from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_translate_endpoint():
    response = client.post(
        "/translate/",
        json={
            "text": "Ndewo",
            "source_language": "igbo",
            "target_language": "english",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["input"] == "Ndewo"
    assert payload["translation"]
