"""A failed or corrupted model install must not redefine trusted checksums."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/download-task-models.py'
    spec = importlib.util.spec_from_file_location('task_download', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def test_download_integrity_and_reuse(downloader, monkeypatch):
    entry = dict(provider='sonic', repo='test/model', revision='pinned',
                 file='model.onnx', bytes=5, sha256=hashlib.sha256(b'model').hexdigest())
    path = downloader.ROOT / 'external/task-models/sonic/model.onnx'
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'wrong'))
    with pytest.raises(ValueError, match='Downloaded asset checksum mismatch'):
        downloader.fetch(entry)
    assert not path.exists() and not path.with_suffix('.onnx.part').exists()
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'model'))
    downloader.fetch(entry)
    assert path.read_bytes() == b'model'
    def no_network(*a, **kw):
        raise AssertionError('Verified cached assets must not be downloaded again')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', no_network)
    downloader.fetch(entry)
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Existing asset checksum mismatch'):
        downloader.fetch(entry)
    assert path.read_bytes() == b'wrong'
