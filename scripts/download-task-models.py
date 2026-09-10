"""Download public inference assets against the committed, immutable hash manifest."""
import concurrent.futures
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def matches(path, entry):
    if path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def fetch(entry):
    folder = ROOT / 'external' / 'task-models' / entry['provider']
    if entry['provider'] == 'ardy':
        folder /= 'ARDY-G1-RP-25FPS-Horizon52'
    path = folder / entry['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not matches(path, entry):
            raise ValueError(f'Existing asset checksum mismatch: {path}. Move it aside and retry.')
        return
    partial = path.with_suffix(path.suffix + '.part')
    try:
        url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
        print('Downloading', entry['provider'], entry['file'], flush=True)
        with urllib.request.urlopen(url, timeout=90) as src, partial.open('wb') as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        if not matches(partial, entry):
            raise ValueError(f'Downloaded asset checksum mismatch: {path}')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'integrations/task-models.manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, manifest))
    print('Verified', len(manifest), 'pinned assets', flush=True)
