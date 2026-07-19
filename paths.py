"""paths.py — Single source of truth for every Drive / repo / version path.

Both the local Windows path (``G:/My Drive/MusicProject``) and the Colab
mount (``/content/drive/MyDrive/MusicProject``) are supported through the
same class so notebooks become 3-line imports:

    from paths import DrivePaths
    P = DrivePaths.colab(version_name='Israeli_Shalom_Arik')   # or DrivePaths.local(...)
    print(P.splits_dir, P.checkpoints_dir)

Every constant previously scattered across notebooks lives here. If a path
needs to change (e.g. Drive folder renamed), edit ONE file.

Conventions
-----------
- ``version_name`` matches the ``style_name``-derived folder under
  ``versions/`` and ``checkpoints/`` (e.g. ``Israeli_Shalom_Arik``).
- Slakh tensors live at the project root under ``slakh_processed/slakh`` and
  splits at ``data/slakh_splits/`` — these are FIXED, not versioned.
- Israeli (and any future style) lives under ``versions/<version_name>/`` —
  fully self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repo root = directory containing this file.
REPO_ROOT: Path = Path(__file__).resolve().parent

# Default Drive roots for each surface. Override via ``DrivePaths(drive_root=...)``.
DRIVE_ROOT_LOCAL = Path(r"G:/My Drive/MusicProject")
DRIVE_ROOT_COLAB = Path("/content/drive/MyDrive/MusicProject")


@dataclass(frozen=True)
class DrivePaths:
    """All paths used by ingest / training / postprocessing notebooks.

    Build via the ``local`` or ``colab`` classmethods rather than __init__
    so the surface auto-resolves.
    """

    drive_root: Path
    version_name: str = "Israeli_Shalom_Arik"

    # ---- alternate constructors -------------------------------------------------
    @classmethod
    def local(cls, version_name: str = "Israeli_Shalom_Arik") -> "DrivePaths":
        return cls(drive_root=DRIVE_ROOT_LOCAL, version_name=version_name)

    @classmethod
    def colab(cls, version_name: str = "Israeli_Shalom_Arik") -> "DrivePaths":
        return cls(drive_root=DRIVE_ROOT_COLAB, version_name=version_name)

    @classmethod
    def auto(cls, version_name: str = "Israeli_Shalom_Arik") -> "DrivePaths":
        """Pick Colab if /content/drive exists, else local Windows Drive."""
        if DRIVE_ROOT_COLAB.parent.exists():
            return cls.colab(version_name)
        return cls.local(version_name)

    # ---- repo-side constants ----------------------------------------------------
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def version_spec(self) -> Path:
        return REPO_ROOT / "configs" / f"version_{self.version_name}.yaml"

    @property
    def run_spec(self) -> Path:
        return REPO_ROOT / "configs" / f"run_spec_{self.version_name}.yaml"

    @property
    def default_config(self) -> Path:
        return REPO_ROOT / "configs" / "default.yaml"

    # ---- pool / version roots ---------------------------------------------------
    @property
    def source_pool(self) -> Path:
        return self.drive_root / "SourcePool"

    @property
    def source_pool_index(self) -> Path:
        return self.drive_root / "SourcePool" / "source_pool_index.csv"

    @property
    def version_dir(self) -> Path:
        return self.drive_root / "versions" / self.version_name

    @property
    def processed_data_dir(self) -> Path:
        return self.version_dir / "processed_data"

    @property
    def manifest_csv(self) -> Path:
        """All segments (train pool + held-out) produced by derive_version."""
        return self.version_dir / "manifest.csv"

    @property
    def train_pool_manifest(self) -> Path:
        """Manifest minus held-out songs — input to split_dataset."""
        return self.version_dir / "train_pool_manifest.csv"

    @property
    def held_out_manifest(self) -> Path:
        return self.version_dir / "held_out.csv"

    @property
    def splits_dir(self) -> Path:
        return self.version_dir / "splits"

    @property
    def train_csv(self) -> Path:
        return self.splits_dir / "train.csv"

    @property
    def val_csv(self) -> Path:
        return self.splits_dir / "val.csv"

    @property
    def test_csv(self) -> Path:
        return self.splits_dir / "test.csv"

    @property
    def lock_file(self) -> Path:
        """Reproducibility lock — see paths.write_lock_file / verify_lock_file."""
        return self.version_dir / f"version_{self.version_name}.lock.json"

    # ---- Slakh (fixed, version_id=0) --------------------------------------------
    @property
    def slakh_processed(self) -> Path:
        return self.drive_root / "slakh_processed"

    @property
    def slakh_manifest(self) -> Path:
        return self.drive_root / "data" / "slakh_manifest.csv"

    @property
    def slakh_splits_dir(self) -> Path:
        return self.drive_root / "data" / "slakh_splits"

    @property
    def slakh_train_csv(self) -> Path:
        return self.slakh_splits_dir / "train.csv"

    @property
    def slakh_val_csv(self) -> Path:
        return self.slakh_splits_dir / "val.csv"

    # ---- combined (multi-style) manifests ---------------------------------------
    @property
    def combined_train_csv(self) -> Path:
        return self.version_dir / "combined_train.csv"

    @property
    def combined_val_csv(self) -> Path:
        return self.version_dir / "combined_val.csv"

    # ---- training outputs -------------------------------------------------------
    @property
    def checkpoints_dir(self) -> Path:
        return self.drive_root / "checkpoints" / self.version_name

    @property
    def logs_dir(self) -> Path:
        return self.drive_root / "logs" / self.version_name

    @property
    def slakh_ckpt(self) -> Path:
        """Optional warm-start checkpoint from the Slakh sanity run."""
        return self.drive_root / "checkpoints" / "slakh_sanity" / "best_val.pt"

    # ---- inference / deliverables ----------------------------------------------
    @property
    def inference_runs_dir(self) -> Path:
        return self.version_dir / "inference_runs"

    # ---- helpers ----------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create all output directories used by ingest + training."""
        for d in [
            self.version_dir,
            self.splits_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.inference_runs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return (
            f"DrivePaths(version={self.version_name})\n"
            f"  drive_root      : {self.drive_root}\n"
            f"  source_pool     : {self.source_pool}\n"
            f"  version_dir     : {self.version_dir}\n"
            f"  splits_dir      : {self.splits_dir}\n"
            f"  checkpoints_dir : {self.checkpoints_dir}\n"
            f"  logs_dir        : {self.logs_dir}\n"
        )


# ============================================================================
# Reproducibility lock-file API
# ============================================================================
def write_lock_file(
    paths: DrivePaths,
    *,
    derive_summary: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Pin a version to its inputs.

    Writes JSON with:
      - git commit + dirty flag
      - SHA256 of configs/version_<v>.yaml + configs/default.yaml
      - segment / hour totals (from derive_summary.json if available)
      - datetime UTC

    Returns the lock-file path.
    """
    import hashlib
    import json
    import subprocess
    from datetime import datetime, timezone

    def _sha256(p: Path) -> str | None:
        if not p.is_file():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip()
        except Exception:
            return None

    lock = {
        "version_name": paths.version_name,
        "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": {
            "version_spec": _sha256(paths.version_spec),
            "default_config": _sha256(paths.default_config),
        },
        "derive_summary": derive_summary,
        "extra": extra or {},
    }

    paths.lock_file.parent.mkdir(parents=True, exist_ok=True)
    paths.lock_file.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return paths.lock_file


def verify_lock_file(paths: DrivePaths, *, strict: bool = True) -> dict:
    """Verify that the current repo + configs match the lock-file on disk.

    Raises RuntimeError on mismatch when ``strict=True``; otherwise returns
    a dict describing the comparison so the caller can decide.
    """
    import hashlib
    import json
    import subprocess

    if not paths.lock_file.is_file():
        raise FileNotFoundError(
            f"No lock file at {paths.lock_file}. Run the data ingest notebook first."
        )

    lock = json.loads(paths.lock_file.read_text(encoding="utf-8"))

    def _sha256(p: Path) -> str | None:
        if not p.is_file():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    current = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": {
            "version_spec": _sha256(paths.version_spec),
            "default_config": _sha256(paths.default_config),
        },
    }

    # Data-processing drift is determined by the version_spec ONLY: the per-version
    # YAML owns the song list + the DSP params (sample_rate / hop / n_fft / n_mels /
    # segment_duration) that actually produce the tensors → a version_spec hash
    # change is FATAL under strict mode.
    #
    # default_config (configs/default.yaml) holds MODEL + TRAINING hyperparameters
    # (LR schedule, total_steps, batch_size, architecture) and shape descriptors —
    # none of which change the tensorized data. A default_config drift is therefore
    # a soft WARNING, not a hard failure. Likewise a git_commit mismatch with the
    # version_spec hash intact is code/notebook-only drift → soft WARNING.
    spec_mismatch = (
        current["config_sha256"]["version_spec"]
        != lock["config_sha256"].get("version_spec")
    )
    default_cfg_mismatch = (
        current["config_sha256"]["default_config"]
        != lock["config_sha256"].get("default_config")
    )

    fatal_mismatches = []
    if spec_mismatch:
        fatal_mismatches.append("config_sha256[version_spec] changed since ingest")

    soft_mismatches = []
    if default_cfg_mismatch:
        soft_mismatches.append(
            "config_sha256[default_config] changed since ingest "
            "(model/training hyperparams — tensors unaffected)"
        )

    commit_mismatch = current["git_commit"] != lock.get("git_commit")
    commit_msg = (
        f"git_commit: lock={lock.get('git_commit')[:8] if lock.get('git_commit') else None} "
        f"current={current['git_commit'][:8] if current['git_commit'] else None}"
    )

    mismatches = list(fatal_mismatches) + list(soft_mismatches)
    if commit_mismatch:
        mismatches.append(commit_msg)

    result = {"lock": lock, "current": current, "mismatches": mismatches}

    if fatal_mismatches and strict:
        msg = (
            "Lock-file verification FAILED — data inputs drifted since ingest:\n  - "
            + "\n  - ".join(fatal_mismatches)
            + f"\n\nLock file: {paths.lock_file}"
            "\n\nEither (a) re-run the ingest notebook to refresh the lock,"
            "\nor (b) git-checkout the commit recorded in the lock before training."
        )
        raise RuntimeError(msg)

    if soft_mismatches:
        # default.yaml changed but the version_spec (data contract) is byte-identical.
        print(
            "  ⚠ " + "; ".join(soft_mismatches) + ".\n"
            "    Treating as training-config drift (tensors unchanged). Re-run the "
            "ingest lock cell to silence this."
        )

    if commit_mismatch:
        # Code/notebooks moved but the version_spec is byte-identical to ingest.
        print(
            "  ⚠ git commit differs from lock but version_spec hash matches "
            f"({commit_msg}).\n"
            "    Treating as code-only drift (data unchanged). Re-run ingest "
            "lock cell to silence this."
        )

    return result


# ============================================================================
# Environment guard — assert the right Python is active
# ============================================================================
def assert_env(*, expect: str) -> None:
    """Assert the running interpreter belongs to the expected environment.

    ``expect`` is a substring matched against ``sys.executable``. Typical
    values: ``"ml_env"`` (local CUDA stack), ``"basic_pitch_env"`` (local
    Basic-Pitch / TF stack), ``"colab"`` or ``"/usr/"`` (Colab runtime).
    """
    import sys

    exe = sys.executable.lower().replace("\\", "/")
    if expect.lower() not in exe:
        raise RuntimeError(
            f"Wrong Python environment.\n"
            f"  Expected substring: {expect!r}\n"
            f"  Active executable : {sys.executable}\n"
            f"  Activate the right env, then restart the kernel."
        )
