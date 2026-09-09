"""Selection groups are workspace presets, independent of any motion project."""
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from .robot import HANDLES, ROOT


class GroupStore:
    def __init__(self, path=ROOT / 'presets/groups.json'):
        self.path = Path(path)
        self.lock = threading.RLock()

    @staticmethod
    def validate(name, members):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
            raise ValueError('프리셋 이름을 1–60자로 입력하세요.')
        if not isinstance(members, list) or not 1 <= len(members) <= len(HANDLES):
            raise ValueError('프리셋에 최소 한 부위를 선택하세요.')
        if any(not isinstance(k, str) or k not in HANDLES for k in members) or len(set(members)) != len(members):
            raise ValueError('알 수 없거나 중복된 부위가 있습니다.')

    def list(self):
        with self.lock:
            if not self.path.exists():
                return []
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('format') != 'motioncreator.groups.v1' or not isinstance(data.get('groups'), list):
                raise ValueError('그룹 프리셋 파일 형식이 올바르지 않습니다.')
            for group in data['groups']:
                self.validate(group['name'], group['members'])
                if not isinstance(group.get('id'), str):
                    raise ValueError('그룹 프리셋 ID가 올바르지 않습니다.')
            return data['groups']

    def _write(self, groups):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.path.parent, encoding='utf-8', delete=False) as f:
                temporary = f.name
                json.dump({'format': 'motioncreator.groups.v1', 'groups': groups}, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, name, members, group_id=None):
        self.validate(name, members)
        name = name.strip()
        with self.lock:
            groups = self.list()
            if group_id is not None and not any(g['id'] == group_id for g in groups):
                raise ValueError('수정할 프리셋을 찾을 수 없습니다.')
            if any(g['name'].casefold() == name.casefold() and g['id'] != group_id for g in groups):
                raise ValueError('같은 이름의 프리셋이 있습니다. 다른 이름을 쓰거나 기존 프리셋을 수정하세요.')
            if group_id is None and len(groups) >= 100:
                raise ValueError('프리셋은 최대 100개까지 저장할 수 있습니다.')
            group = {'id': group_id or uuid.uuid4().hex, 'name': name, 'members': members}
            groups = [group if g['id'] == group_id else g for g in groups] if group_id else [*groups, group]
            self._write(groups)
            return group

    def delete(self, group_id):
        with self.lock:
            groups = self.list()
            remaining = [g for g in groups if g['id'] != group_id]
            if len(remaining) == len(groups):
                raise ValueError('삭제할 프리셋을 찾을 수 없습니다.')
            self._write(remaining)
