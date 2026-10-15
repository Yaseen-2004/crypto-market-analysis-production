from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_contract():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_overview_has_coin_list():
    response = client.get("/api/overview")
    assert response.status_code == 200
    payload = response.json()
    assert "coins" in payload
    assert payload["coin_count"] > 0

def test_coin_history():
    overview = client.get("/api/overview").json()
    coin_id = overview["coins"][0]["coin_id"]
    response = client.get(f"/api/coins/{coin_id}/history?limit=5")
    assert response.status_code == 200
    assert len(response.json()["items"]) <= 5
