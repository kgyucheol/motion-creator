"""Pin and verify inference assets separately from reference/scene identities."""
import hashlib,json
from .robot import ROOT

def sha256(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def verify_assets(provider):
    manifest=json.loads((ROOT/'integrations/task-models.manifest.json').read_text())
    entries=[x for x in manifest if x['provider']==provider]
    if not entries: raise ValueError('Missing model asset manifest')
    folder=ROOT/'external/task-models'/provider
    if provider=='ardy': folder=folder/'ARDY-G1-RP-25FPS-Horizon52'
    for entry in entries:
        path=folder/entry['file']
        if not path.is_file() or sha256(path)!=entry['sha256']:
            raise ValueError(f'{provider} model asset checksum mismatch: {entry["file"]}')
    return entries
