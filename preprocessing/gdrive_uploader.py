"""
Google Drive Upload Script
===========================
This script automatically uploads files to Google Drive and can be integrated
with the YouTube downloader to sync downloaded audio files to the cloud.
"""

import os
import pickle
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


# Define the scopes required for Google Drive access
# SCOPES determine what permissions the app needs
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def authenticate_google_drive():
    """
    Authenticates the user with Google Drive API and returns the service object.
    
    This function handles the OAuth 2.0 authentication flow:
    1. Checks if valid credentials already exist (token.pickle)
    2. If not, or if expired, initiates browser-based authentication
    3. Saves credentials for future use
    
    Returns:
    --------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    Raises:
    -------
    Exception
        If authentication fails or credentials.json is missing
    
    Notes:
    ------
    First-time setup requires:
    1. Go to Google Cloud Console: https://console.cloud.google.com/
    2. Create a project and enable Google Drive API
    3. Create OAuth 2.0 credentials (Desktop app)
    4. Download credentials.json and place in project folder
    """
    creds = None
    
    # Token file stores user's access and refresh tokens
    # Created automatically after successful authentication
    token_path = 'token.pickle'
    
    # Check if we have saved credentials from a previous run
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate the user
    if not creds or not creds.valid:
        # Try to refresh expired credentials
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            # No credentials exist - need to authenticate
            print("No valid credentials found. Starting authentication flow...")
            print("This will open your browser for Google authentication.")
            
            # Check if credentials.json exists
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "credentials.json not found!\n"
                    "Please follow these steps:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Create a project and enable Google Drive API\n"
                    "3. Create OAuth 2.0 credentials (Desktop app)\n"
                    "4. Download credentials.json to this folder"
                )
            
            # Initiate OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future runs
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        print("✓ Authentication successful! Credentials saved.")
    
    # Build and return the Drive service
    service = build('drive', 'v3', credentials=creds)
    return service


def find_or_create_nested_folder(service, folder_path):
    """
    Finds or creates a nested folder structure in Google Drive.
    
    Parameters:
    -----------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    folder_path : str
        Folder path (e.g., "MusicProject/youtube_downloads")
    
    Returns:
    --------
    str
        The ID of the final (deepest) folder
    """
    # Split path into parts
    folders = folder_path.strip('/').split('/')
    
    parent_id = None
    for folder_name in folders:
        parent_id = find_or_create_folder(service, folder_name, parent_id)
    
    return parent_id


def find_or_create_folder(service, folder_name, parent_id=None):
    """
    Finds an existing folder in Google Drive or creates it if it doesn't exist.
    
    Parameters:
    -----------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    folder_name : str
        Name of the folder to find or create
    
    parent_id : str, optional
        ID of parent folder. If None, creates in root directory
    
    Returns:
    --------
    str
        The folder ID of the found or newly created folder
    """
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    try:
        # Execute search query
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        items = results.get('files', [])
        
        # If folder exists, return its ID
        if items:
            print(f"✓ Found existing folder: '{folder_name}'")
            return items[0]['id']
        
        # Folder doesn't exist - create it
        print(f"Creating new folder: '{folder_name}'")
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        # Set parent if provided
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        # Create the folder
        folder = service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        print(f"✓ Created folder with ID: {folder.get('id')}")
        return folder.get('id')
    
    except HttpError as error:
        print(f"✗ Error finding/creating folder: {error}")
        raise


def upload_file_to_drive(service, file_path, folder_id=None):
    """
    Uploads a file to Google Drive.
    
    Parameters:
    -----------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    file_path : str or Path
        Path to the file to upload
    
    folder_id : str, optional
        ID of the folder to upload to. If None, uploads to root
    
    Returns:
    --------
    dict
        Information about the uploaded file (id, name, webViewLink)
    
    Raises:
    -------
    Exception
        If upload fails
    """
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Uploading: {file_path.name}")
    print(f"  Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Prepare file metadata
    file_metadata = {
        'name': file_path.name
    }
    
    # Set parent folder if provided
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    # Determine MIME type based on file extension
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.mid': 'audio/midi',
        '.midi': 'audio/midi',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
    }
    mime_type = mime_types.get(file_path.suffix.lower(), 'application/octet-stream')
    
    try:
        # Create media upload object
        media = MediaFileUpload(
            str(file_path),
            mimetype=mime_type,
            resumable=True  # Enable resumable upload for large files
        )
        
        # Upload the file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        print(f"✓ Uploaded successfully!")
        print(f"  File ID: {file.get('id')}")
        print(f"  View: {file.get('webViewLink')}")
        
        return file
    
    except HttpError as error:
        print(f"✗ Upload failed: {error}")
        raise


def upload_directory_to_drive(service, directory_path, folder_name=None, parent_folder=None):
    """
    Uploads all files from a local directory to Google Drive folder.
    
    Parameters:
    -----------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    directory_path : str or Path
        Path to the local directory containing files to upload
    
    folder_name : str, optional
        Name for the Google Drive folder. If None, uses directory name
    
    parent_folder : str, optional
        Name of parent folder to create folder inside. If None, creates in root
    
    Returns:
    --------
    tuple
        (list of uploaded files, folder_id, parent_folder_id)
    """
    directory_path = Path(directory_path)
    
    # Validate directory exists
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Use directory name if folder name not specified
    if folder_name is None:
        folder_name = directory_path.name
    
    # Create parent folder if specified
    parent_folder_id = None
    if parent_folder:
        parent_folder_id = find_or_create_nested_folder(service, parent_folder)
        print(f"Using parent folder: '{parent_folder}'")
    
    # Find or create the target folder in Drive (inside parent if specified)
    folder_id = find_or_create_folder(service, folder_name, parent_folder_id)
    
    # Get all files in directory
    files = [f for f in directory_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in {directory_path}")
        return []
    
    print(f"\nUploading {len(files)} file(s) to Google Drive folder '{folder_name}'")
    print("=" * 60)
    
    uploaded_files = []
    
    # Upload each file
    for idx, file_path in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
        try:
            uploaded_file = upload_file_to_drive(service, file_path, folder_id)
            uploaded_files.append(uploaded_file)
        except Exception as e:
            print(f"✗ Failed to upload {file_path.name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Upload complete: {len(uploaded_files)}/{len(files)} files uploaded")
    
    return uploaded_files, folder_id, parent_folder_id


def share_folder_with_user(service, folder_id, email_address, role='writer'):
    """
    Shares a Google Drive folder with another user.
    
    Parameters:
    -----------
    service : googleapiclient.discovery.Resource
        Authenticated Google Drive API service object
    
    folder_id : str
        ID of the folder to share
    
    email_address : str
        Email address of the user to share with
    
    role : str, optional (default='writer')
        Permission level: 'reader' (view only), 'writer' (can edit), 'commenter'
    
    Returns:
    --------
    dict
        Information about the created permission
    """
    print(f"Sharing folder with: {email_address} (role: {role})")
    
    try:
        # Create permission
        permission = {
            'type': 'user',
            'role': role,
            'emailAddress': email_address
        }
        
        # Apply permission to folder
        result = service.permissions().create(
            fileId=folder_id,
            body=permission,
            sendNotificationEmail=True,  # Send email notification
            fields='id'
        ).execute()
        
        print(f"✓ Folder shared successfully!")
        return result
    
    except HttpError as error:
        print(f"✗ Failed to share folder: {error}")
        raise


def main():
    """
    Main function for command-line usage.
    Demonstrates uploading files and sharing folders.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload files to Google Drive and share folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single file
  python gdrive_uploader.py file.wav
  
  # Upload entire directory
  python gdrive_uploader.py youtube_downloads/ --folder "Music Downloads"
  
  # Upload and share with partner
  python gdrive_uploader.py youtube_downloads/ --share partner@email.com
        """
    )
    
    parser.add_argument(
        'path',
        type=str,
        help='Path to file or directory to upload'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Google Drive folder name (default: uses directory name)'
    )
    
    parser.add_argument(
        '--parent',
        type=str,
        default=None,
        help='Parent folder name to create subfolder inside'
    )
    
    parser.add_argument(
        '--share',
        type=str,
        default=None,
        help='Email address to share the folder with'
    )
    
    parser.add_argument(
        '--share-parent',
        action='store_true',
        help='Share the parent folder instead of the uploaded folder'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        default='writer',
        choices=['reader', 'writer', 'commenter'],
        help='Permission level for shared user (default: writer)'
    )
    
    args = parser.parse_args()
    
    # Display script header
    print("=" * 60)
    print("Google Drive Uploader")
    print("=" * 60)
    print(f"URLs to upload: {args.path}")
    print(f"Parent folder: {args.parent if args.parent else 'None (root)'}")
    print(f"Folder name: {args.folder if args.folder else 'Auto'}")
    print("=" * 60)
    
    try:
        # Authenticate
        service = authenticate_google_drive()
        
        # Convert path to Path object
        path = Path(args.path).resolve()
        
        # Handle parent folder path if specified
        parent_folder_id = None
        if args.parent:
            parent_folder_id = find_or_create_nested_folder(service, args.parent)
        
        # Upload file or directory
        if path.is_file():
            upload_file_to_drive(service, path, parent_folder_id)
        elif path.is_dir():
            uploaded_files, folder_id, parent_folder_id = upload_directory_to_drive(
                service, path, args.folder, args.parent
            )
            
            # Share folder if email provided
            if args.share and uploaded_files:
                # Determine which folder to share
                if args.share_parent and parent_folder_id:
                    # Share the parent folder
                    share_folder_id = parent_folder_id
                    share_name = args.parent
                else:
                    # Share the uploaded folder
                    share_folder_id = folder_id
                    share_name = args.folder or path.name
                
                print(f"\nSharing folder: '{share_name}'")
                share_folder_with_user(
                    service,
                    share_folder_id,
                    args.share,
                    args.role
                )
        else:
            print(f"✗ Path not found: {path}")
            exit(1)
        
        print("\n" + "=" * 60)
        print("Operation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
