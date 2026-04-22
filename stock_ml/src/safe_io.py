"""
Safe I/O module — fixes UnicodeEncodeError on Windows console (cp1252).

Root cause: Python print() uses sys.stdout encoding (cp1252 on Windows),
which cannot encode emoji characters like U+1F504 (🔄), U+2705 (✅), etc.

Solution: Reconfigure sys.stdout and sys.stderr to use UTF-8 with 'replace'
error handling, so unencodable characters are replaced with '?' instead of
raising UnicodeEncodeError.

Usage: Import this module ONCE at the top of any entry-point script:
    import src.safe_io  # noqa: F401  — auto-configures stdout/stderr
"""
import sys
import io
import os


def configure_safe_output():
    """
    Reconfigure stdout/stderr to handle Unicode safely on Windows.
    
    Strategy:
    1. Try to set UTF-8 encoding (best: all characters preserved)
    2. Fallback: wrap with 'replace' error handler (emoji -> '?')
    """
    # Method 1: Set PYTHONIOENCODING environment variable for subprocesses
    if "PYTHONIOENCODING" not in os.environ:
        os.environ["PYTHONIOENCODING"] = "utf-8"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if stream is None:
            continue

        # Already UTF-8 with error handling? Skip.
        if (hasattr(stream, 'encoding') 
            and stream.encoding 
            and stream.encoding.lower().replace('-', '') == 'utf8'
            and hasattr(stream, 'errors')
            and stream.errors in ('replace', 'backslashreplace', 'xmlcharrefreplace')):
            continue

        try:
            # Python 3.7+: use reconfigure() if available
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
                continue
        except Exception:
            pass

        try:
            # Fallback: wrap the underlying buffer with a new TextIOWrapper
            if hasattr(stream, 'buffer'):
                new_stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=stream.line_buffering if hasattr(stream, 'line_buffering') else True,
                )
                setattr(sys, stream_name, new_stream)
        except Exception:
            pass


# Auto-configure on import
configure_safe_output()
