"""
Google Drive Sync Utilities
============================

Supports Path B of the dual-execution design:
  Local preprocessing  →  upload_song_to_drive()  →  clean_local_song()

This is the current Israeli ingest path: local code creates all artifacts, this
module mirrors the song folder to Drive, and only then may the local copy be
deleted.

The caller (batch_ingest.py or a Colab cell) controls whether and when to
upload and clean; process_song() itself has no Drive dependency.

Authentication:
  Uses credentials.json (OAuth2 client secrets) in the project root.
  On first run it opens a browser for consent and caches the token in
  token.pickle next to credentials.json.

Usage example (local → Drive → clean):
    from pathlib import Path
    from preprocessing.drive_sync import upload_song_to_drive, clean_local_song

    result = process_song(row, out_root=Path("/tmp/local_data"))
    if result["status"] == "ok":
        upload_song_to_drive(
            song_dir=Path(result["song_dir"]),
            drive_root_id="<your Drive folder ID>",
        )
        clean_local_song(Path(result["song_dir"]))
"""

import os
import pickle
import shutil
import ssl
import time
from pathlib import Path
from typing import Optional

# Google API imports (google-api-python-client, google-auth-oauthlib)
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_CREDENTIALS_FILE = Path(__file__).parent.parent / "credentials.json"
_TOKEN_FILE       = Path(__file__).parent.parent / "token.pickle"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_drive_service():
    """Return an authenticated Google Drive API service object."""
    creds = None
    if _TOKEN_FILE.exists():
        with open(_TOKEN_FILE, "rb") as fh:
            creds = pickle.load(fh)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Reuse the cached OAuth refresh token when possible; this avoids a
            # browser prompt on every batch ingest run.
            creds.refresh(Request())
        else:
            # First run on a machine: open the browser consent flow and cache
            # the resulting token for future uploads.
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_FILE), _SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(_TOKEN_FILE, "wb") as fh:
            pickle.dump(creds, fh)

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------

def _get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return Drive folder ID for *name* under *parent_id*, creating if absent."""
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def _ensure_drive_path(service, path_parts: list[str], root_id: str) -> str:
    """Walk/create a sequence of nested folders under *root_id*.

    Example: ``_ensure_drive_path(svc, ["Artist", "Album", "Song"], root_id)``
    """
    current_id = root_id
    for part in path_parts:
        current_id = _get_or_create_folder(service, part, current_id)
    return current_id


def _file_exists_on_drive(service, name: str, parent_id: str) -> bool:
    """Return True if a file with *name* already exists under *parent_id* on Drive."""
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id)").execute()
    return len(results.get("files", [])) > 0


def _upload_file(service, local_path: Path, parent_id: str, max_retries: int = 5) -> str:
    """Upload *local_path* to Drive under *parent_id* with retry on transient errors.

    Skips upload if the file already exists on Drive (resume support).
    Retries on SSLEOFError and ConnectionError with exponential backoff.
    """
    if _file_exists_on_drive(service, local_path.name, parent_id):
        return "(skipped — already on Drive)"

    media = MediaFileUpload(str(local_path), resumable=True)
    file_meta = {"name": local_path.name, "parents": [parent_id]}

    for attempt in range(max_retries):
        try:
            uploaded = service.files().create(
                body=file_meta, media_body=media, fields="id"
            ).execute()
            return uploaded["id"]
        except (ssl.SSLEOFError, ssl.SSLError, ConnectionError, OSError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 s
            print(f"  [retry {attempt+1}/{max_retries-1}] {exc.__class__.__name__}: {exc} — retrying in {wait}s")
            time.sleep(wait)
            # Re-create media object for retry
            media = MediaFileUpload(str(local_path), resumable=True)

    raise RuntimeError(f"Failed to upload {local_path} after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Public API — setup helpers
# ---------------------------------------------------------------------------

def get_or_create_music_data_folder(
    path: str = "MusicProject/MusicProjectData",
    verbose: bool = True,
) -> str:
    """Return the Drive folder ID for MusicProjectData, creating the full path if needed.

    Walks from the Drive root ('My Drive') through each path component,
    creating any missing folders.  On success prints the folder ID so you can
    record it for future runs.

    Parameters
    ----------
    path : str
        Slash-separated path of folders to create/find under My Drive root.
        Default: ``"MusicProject/MusicProjectData"``

    Returns
    -------
    str
        Drive folder ID of the last folder in the path.
    """
    service = _get_drive_service()

    # "root" is the special alias for My Drive in the Drive API
    folder_id = _ensure_drive_path(service, path.split("/"), "root")

    if verbose:
        print(f"Drive folder ID for '{path}': {folder_id}")
        print("(Save this ID — pass it as drive_music_data_id in upload_song_to_drive)")
    return folder_id


# ---------------------------------------------------------------------------
# Public API — data transfer
# ---------------------------------------------------------------------------

def upload_song_to_drive(
    song_dir: Path,
    drive_music_data_id: str,
    verbose: bool = True,
) -> None:
    """Recursively upload *song_dir* to Drive under MusicProjectData/.

    The folder structure  ``{artist}/{album}/{song_name}/``  is mirrored on
    Drive under the folder whose ID is *drive_music_data_id*.

    Parameters
    ----------
    song_dir : Path
        The local song output folder produced by process_song().
        Expected depth:  MusicProjectData/{artist}/{album}/{song_name}/
    drive_music_data_id : str
        Drive folder ID of the remote MusicProjectData/ folder.
    verbose : bool
        Print progress per file.
    """
    service = _get_drive_service()

    # Reconstruct  [artist, album, song_name]  from the last three path parts
    parts = song_dir.parts[-3:]
    remote_song_folder_id = _ensure_drive_path(service, list(parts), drive_music_data_id)

    def _upload_tree(local: Path, drive_parent_id: str):
        """Upload one local directory subtree into a Drive folder."""
        for item in sorted(local.iterdir()):
            if item.is_dir():
                # Mirror the local folder tree exactly so Colab paths and local
                # paths describe the same artist/album/song structure.
                child_id = _get_or_create_folder(service, item.name, drive_parent_id)
                _upload_tree(item, child_id)
            else:
                # _upload_file handles resume-by-name, so repeated uploads are
                # safe after an interrupted run or a skipped song.
                result = _upload_file(service, item, drive_parent_id)
                if verbose:
                    tag = "Skipped" if result == "(skipped — already on Drive)" else "Uploaded"
                    print(f"  {tag}: {item.relative_to(song_dir.parent.parent.parent)}")

    _upload_tree(song_dir, remote_song_folder_id)
    if verbose:
        print(f"✓ Upload complete: {song_dir.name} → Drive")


def clean_local_song(song_dir: Path, verbose: bool = True) -> None:
    """Delete *song_dir* from local disk.

    Only call this after confirming a successful upload.
    """
    if song_dir.exists():
        shutil.rmtree(song_dir)
        if verbose:
            print(f"✓ Cleaned local: {song_dir}")
    else:
        if verbose:
            print(f"  Already absent: {song_dir}")
