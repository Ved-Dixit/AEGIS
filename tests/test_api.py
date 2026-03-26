from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sources_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/sources")
    assert response.status_code == 200
    payload = response.json()
    assert any(source["provider"] == "Kaggle" for source in payload)
    assert any(source["provider"] == "GitHub" for source in payload)


def test_public_sources_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/sources/public")
    assert response.status_code == 200
    payload = response.json()
    assert any(source["access_mode"] == "direct_download" for source in payload)
    assert any(source["access_mode"] == "kaggle_cli" for source in payload)
