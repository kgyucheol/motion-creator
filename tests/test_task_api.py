from fastapi.testclient import TestClient
from motioncreator.server import app
from motioncreator import task_jobs


def test_task_api_validation_persistence_and_missing_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(task_jobs,'TASKS',tmp_path)
    with TestClient(app) as client:
        defaults=client.get('/api/tasks/defaults').json()
        assert defaults['actual']['size'][1]==.51 and defaults['perceived']['size'][1]==.49
        response=client.post('/api/tasks',json={'spec':defaults})
        assert response.status_code==200
        task=response.json();tid=task['id']
        assert client.get('/api/tasks/'+tid).json()==task
        assert client.get('/api/tasks').json()[0]['id']==tid
        assert client.get('/api/tasks/'+tid+'/runs').json()==[]
        assert client.post('/api/tasks/'+tid+'/runs',json={'kind':'simulate'}).status_code==422
        assert client.get('/api/tasks/'+tid+'/runs/012345abcdef').status_code==404
        assert client.get('/api/tasks/runtime').json()['device']=='cpu'
        defaults['actual']['pose']['quaternion_xyzw']=[0,0,0,0]
        assert client.post('/api/tasks',json={'spec':defaults}).status_code==422
