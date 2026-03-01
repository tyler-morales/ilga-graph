"""Shared file validation: magic-byte checks for image uploads.

Used by story images, update images, and bug-report images to reject spoofed
content types (e.g. executable with Content-Type: image/jpeg).
"""

from __future__ import annotations

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_WEBP = b"WEBP"  # at offset 8
_GIF87a = b"GIF87a"
_GIF89a = b"GIF89a"


def magic_matches_image_content_type(content: bytes, content_type: str) -> bool:
    """Return True if file magic bytes match the declared image content type."""
    if not content:
        return False
    if content_type == "image/jpeg":
        return content.startswith(_JPEG_MAGIC)
    if content_type == "image/png":
        return content.startswith(_PNG_MAGIC)
    if content_type == "image/webp":
        return len(content) >= 12 and content.startswith(_WEBP_RIFF) and content[8:12] == _WEBP_WEBP
    if content_type == "image/gif":
        return content.startswith(_GIF87a) or content.startswith(_GIF89a)
    return False
