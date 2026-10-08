import pytest
import app as risk_app

LOW = {"age": 42, "vehicle_age": 2, "annual_km": 8000, "past_claims": 0, "violations": 0, "credit_score": 800}
HIGH = {"age": 20, "vehicle_age": 12, "annual_km": 35000, "past_claims": 4, "violations": 5, "credit_score": 420}
MID = {"age": 35, "vehicle_age": 6, "annual_km": 18000, "past_claims": 1, "violations": 1, "credit_score": 680}


@pytest.fixture
def client(tmp_path):
    risk_app.DB = str(tmp_path / "test.db")
    risk_app.MODEL.clear()
    risk_app.app.config["TESTING"] = True
    risk_app.init_db()
    with risk_app.app.test_client() as c:
        yield c


def login(c):
    return c.post("/", data={"username": "admin", "password": "admin123"}, follow_redirects=True)


def add(c, no, r):
    return c.post("/policyholders", data=dict(r, policy_no=no, vehicle_type="Sedan"))


def test_risk_score_order():
    assert risk_app.risk_score(LOW) < risk_app.risk_score(MID) < risk_app.risk_score(HIGH)
    assert 0 <= risk_app.risk_score(HIGH) <= 100


def test_login_success_and_failure(client):
    assert b"Dashboard" in login(client).data
    bad = client.post("/", data={"username": "x", "password": "y"}, follow_redirects=True)
    assert b"Invalid username or password" in bad.data


def test_protected_page_redirects(client):
    assert client.get("/segment").status_code == 302


def test_segmentation_and_prediction(client):
    login(client)
    for i in range(6):
        for name, r in (("L", LOW), ("M", MID), ("H", HIGH)):
            add(client, f"{name}{i}", {k: v + i % 2 for k, v in r.items()})
    res = client.get("/segment")
    assert res.status_code == 200 and b"Silhouette" in res.data
    assert risk_app.predict_segment(LOW) == "Low Risk"
    assert risk_app.predict_segment(HIGH) == "High Risk"
    out = client.post("/predict", data={k: str(v) for k, v in HIGH.items()})
    assert b"High Risk" in out.data
