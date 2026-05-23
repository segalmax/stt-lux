import hashlib


def file_hash(path: str) -> str:
    """SHA256 of file contents (short hex prefix)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
