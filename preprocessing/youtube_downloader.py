"""
YouTube Audio Downloader Script
================================
This script downloads audio from YouTube videos using yt-dlp library.
Downloads can be saved in various audio formats (WAV, MP3, etc.) for later processing.
"""

import os
import re
import argparse
from pathlib import Path
import yt_dlp


def create_download_directory(output_dir):
    """
    Creates the download directory if it doesn't exist.
    
    Parameters:
    -----------
    output_dir : str or Path
        Path to the download directory
    
    Returns:
    --------
    Path
        Path object pointing to the download directory
    """
    # Convert to Path object for easier handling
    output_dir = Path(output_dir)
    
    # Create directory and any necessary parent directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def _sanitize_name(name: str) -> str:
    """Replace spaces/special chars with underscores for filesystem-safe names."""
    # Remove content in parentheses/brackets (e.g., "(Official Video)")
    name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
    # Replace common separators and whitespace with underscore
    name = re.sub(r'[\s\-–—]+', '_', name.strip())
    # Remove non-alphanumeric chars (keep underscores, Hebrew, etc.)
    name = re.sub(r'[^\w]', '', name, flags=re.UNICODE)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def extract_youtube_metadata(url: str) -> dict:
    """
    Extract artist, album, and song metadata from a YouTube URL.

    Uses yt-dlp to fetch video info and parses the title to infer
    artist and song name.  Typical YouTube music title formats:
        "Artist - Song Title"
        "Artist - Song Title (Official Video)"
        "Song Title | Artist"

    Returns:
        dict with keys: artist, album, song, title (raw)
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    raw_title = info.get('title', 'Unknown')
    # yt-dlp sometimes fills dedicated fields (artist, track, album)
    yt_artist = info.get('artist') or info.get('creator') or info.get('uploader') or ''
    yt_track  = info.get('track') or ''
    yt_album  = info.get('album') or ''

    # --- Fallback: parse title string ---
    parsed_artist = ''
    parsed_song   = raw_title

    # Try "Artist - Song" pattern (most common for music videos)
    for sep in [' - ', ' – ', ' — ', ' | ']:
        if sep in raw_title:
            parts = raw_title.split(sep, 1)
            parsed_artist = parts[0].strip()
            parsed_song   = parts[1].strip()
            break

    # Priority: yt-dlp structured fields > parsed from title > uploader
    artist = _sanitize_name(yt_artist if yt_artist and yt_artist != info.get('uploader', '') else parsed_artist or yt_artist)
    song   = _sanitize_name(yt_track  or parsed_song)
    album  = _sanitize_name(yt_album) if yt_album else 'Singles'

    # Final fallback for artist
    if not artist:
        artist = _sanitize_name(info.get('uploader', 'Unknown_Artist'))

    return {
        'artist': artist,
        'album': album,
        'song': song,
        'title': raw_title,
        'duration': info.get('duration', 0),
    }


def download_youtube_audio(url, output_dir, audio_format='wav', audio_quality='best',
                           cookies_file=None):
    """
    Downloads audio from a YouTube video URL.
    
    This function uses yt-dlp to download and extract audio from YouTube videos.
    The audio is automatically converted to the specified format.
    
    Parameters:
    -----------
    url : str
        YouTube video URL to download audio from
    
    output_dir : str or Path
        Directory where the audio file will be saved
    
    audio_format : str, optional (default='wav')
        Output audio format. Options: 'wav', 'mp3', 'flac', 'aac', 'm4a', 'opus', 'vorbis'
        - 'wav': Uncompressed, best quality, larger files (recommended for MIDI conversion)
        - 'mp3': Compressed, good quality, smaller files
        - 'flac': Lossless compression, high quality, medium file size
    
    audio_quality : str, optional (default='best')
        Audio quality setting. Options: 'best', '320', '256', '192', '128'
        - 'best': Highest available quality
        - Numbers (320, 256, etc.): Bitrate in kbps for compressed formats

    cookies_file : str or None, optional
        Path to a Netscape-format cookies.txt file for authenticated downloads.
        Required on Colab/server environments where YouTube bot-detection blocks
        anonymous requests.  Falls back to the YTDLP_COOKIES_FILE env var if None.
    
    Returns:
    --------
    Path
        Path to the downloaded audio file
    
    Raises:
    -------
    Exception
        If the download fails or URL is invalid
    """
    # Create output directory if it doesn't exist
    output_dir = create_download_directory(output_dir)
    
    # Define output template for downloaded files
    # %(title)s = video title, %(id)s = video ID, %(ext)s = file extension
    output_template = str(output_dir / '%(title)s.%(ext)s')
    
    # Resolve cookies file: explicit arg > env var > None
    if cookies_file is None:
        cookies_file = os.environ.get('YTDLP_COOKIES_FILE')

    # Resolve ffmpeg: use env var or system PATH (skip Windows-local hardcoded path on non-Windows)
    import platform
    _WIN_FFMPEG = r'C:\Users\yotam\ffmpeg-2025-07-07-git-d2828ab284-essentials_build\bin'
    if platform.system() == 'Windows' and Path(_WIN_FFMPEG).exists():
        _ffmpeg_location = _WIN_FFMPEG
    else:
        _ffmpeg_location = None  # rely on system PATH (ffmpeg pre-installed on Colab)

    # Configure yt-dlp options
    ydl_opts = {
        # Output filename template
        'outtmpl': output_template,
        
        # Extract audio only (no video)
        'format': 'bestaudio/best',
        
        # FFmpeg location (None = use system PATH)
        'ffmpeg_location': _ffmpeg_location,
        
        # Post-processing options (convert to desired format)
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
            'preferredcodec': audio_format,  # Convert to specified format
            'preferredquality': audio_quality,  # Set quality level
        }],
        
        # Additional options for better handling
        'noplaylist': True,  # Download only single video (not playlist)
        'quiet': False,  # Show download progress
        'no_warnings': False,  # Show warnings
        'extract_flat': False,  # Extract full info
        
        # Handle errors gracefully
        'ignoreerrors': False,  # Stop on errors
        'no_color': False,  # Allow colored output in terminal
    }

    # Add cookies if provided (bypasses YouTube bot detection on Colab/servers)
    if cookies_file and Path(cookies_file).exists():
        ydl_opts['cookiefile'] = str(cookies_file)
        print(f"Using cookies file: {cookies_file}")
    
    print(f"Downloading audio from: {url}")
    print(f"Output directory: {output_dir}")
    print(f"Audio format: {audio_format}")
    print(f"Quality: {audio_quality}")
    print("-" * 60)
    
    try:
        # Create yt-dlp instance with configured options
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video information first (to get title, duration, etc.)
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            print(f"Video: {video_title}")
            print(f"Duration: {duration // 60}:{duration % 60:02d}")
            print("-" * 60)
            
            # Download and process the audio
            ydl.download([url])
            
            # Construct the expected output file path
            # yt-dlp sanitizes the title and adds the extension
            sanitized_title = ydl.prepare_filename(info)
            output_path = Path(sanitized_title).with_suffix(f'.{audio_format}')
            
            if output_path.exists():
                print(f"\n✓ Successfully downloaded: {output_path.name}")
                print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
                return output_path, info
            else:
                raise Exception(f"Download completed but file not found at expected location")
                
    except Exception as e:
        print(f"\n✗ Error downloading audio: {str(e)}")
        raise


def download_multiple_youtube_audios(urls, output_dir, audio_format='wav', audio_quality='best'):
    """
    Downloads audio from multiple YouTube video URLs.
    
    Parameters:
    -----------
    urls : list of str
        List of YouTube video URLs to download
    
    output_dir : str or Path
        Directory where all audio files will be saved
    
    audio_format : str, optional (default='wav')
        Output audio format for all downloads
    
    audio_quality : str, optional (default='best')
        Audio quality setting for all downloads
    
    Returns:
    --------
    list of Path
        List of paths to successfully downloaded audio files
    """
    downloaded_files = []
    failed_downloads = []
    
    print("=" * 60)
    print(f"Downloading {len(urls)} audio file(s)")
    print("=" * 60)
    
    # Download each URL one by one
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] Processing URL...")
        try:
            # Download single audio file
            output_path = download_youtube_audio(url, output_dir, audio_format, audio_quality)
            downloaded_files.append(output_path)
        except Exception as e:
            # Track failed downloads but continue with others
            print(f"Failed to download: {url}")
            failed_downloads.append((url, str(e)))
        
        print("-" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"✓ Successful: {len(downloaded_files)}")
    print(f"✗ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print("\nFailed URLs:")
        for url, error in failed_downloads:
            print(f"  - {url}")
            print(f"    Error: {error}")
    
    return downloaded_files


def main():
    """
    Main function to handle command-line interface for YouTube audio downloading.
    
    Parses command-line arguments and executes the download process.
    """
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos using yt-dlp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single video as WAV
  python youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  
  # Download single video as MP3 to custom folder
  python youtube_downloader.py "URL" -o downloads -f mp3
  
  # Download multiple videos as WAV
  python youtube_downloader.py "URL1" "URL2" "URL3"
  
  # Download with specific quality
  python youtube_downloader.py "URL" -f mp3 -q 320
  
  # Download to specific folder with high quality WAV
  python youtube_downloader.py "URL" -o youtube_audio -f wav -q best
        """
    )
    
    # Required argument: YouTube URL(s)
    parser.add_argument(
        'urls',
        nargs='+',  # Accept one or more URLs
        type=str,
        help='YouTube video URL(s) to download. Can provide multiple URLs separated by spaces.'
    )
    
    # Optional argument: output directory
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='youtube_downloads',
        help='Output directory for downloaded audio files (default: youtube_downloads)'
    )
    
    # Optional argument: audio format
    parser.add_argument(
        '-f', '--format',
        type=str,
        default='wav',
        choices=['wav', 'mp3', 'flac', 'aac', 'm4a', 'opus', 'vorbis'],
        help='Audio format (default: wav). WAV recommended for best MIDI conversion quality.'
    )
    
    # Optional argument: audio quality
    parser.add_argument(
        '-q', '--quality',
        type=str,
        default='best',
        help='Audio quality: "best" or bitrate like "320", "256", "192", "128" (default: best)'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Display script header
    print("=" * 60)
    print("YouTube Audio Downloader")
    print("=" * 60)
    print(f"URLs to download: {len(args.urls)}")
    print(f"Output directory: {args.output}")
    print(f"Format: {args.format}")
    print(f"Quality: {args.quality}")
    print("=" * 60)
    
    try:
        # Execute the download(s)
        if len(args.urls) == 1:
            # Single URL download
            output_file = download_youtube_audio(
                url=args.urls[0],
                output_dir=args.output,
                audio_format=args.format,
                audio_quality=args.quality
            )
            print("\n" + "=" * 60)
            print("Download completed successfully!")
            print("=" * 60)
        else:
            # Multiple URLs download
            downloaded_files = download_multiple_youtube_audios(
                urls=args.urls,
                output_dir=args.output,
                audio_format=args.format,
                audio_quality=args.quality
            )
            print("=" * 60)
            print("All downloads completed!")
            print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Download failed: {str(e)}")
        print("=" * 60)
        exit(1)


# Entry point of the script
if __name__ == "__main__":
    main()
