"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evaluate_endpoint_smoke_path() -> None:
    csv_payload = (
        "conversation_id,turn_id,speaker,text\n"
        "conv_a,t1,user,Can you help me summarize this note?\n"
        "conv_a,t2,assistant,Yes. Share the note and I will condense it.\n"
    ).encode("utf-8")
    response = client.post(
        "/evaluate",
        files={"file": ("sample.csv", csv_payload, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_metadata"]["facet_count"] == 300
