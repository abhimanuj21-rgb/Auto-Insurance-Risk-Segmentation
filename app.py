from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = "devops-mini-project-secret"
DB = "insurance.db"
FEATURES = ["age", "vehicle_age", "annual_km", "past_claims", "violations", "credit_score"]
SEGMENTS = ["Low Risk", "Medium Risk", "High Risk"]
BASE_PREMIUM = 12000                                   # base premium in Rs
LOADING = {"Low Risk": 0.85, "Medium Risk": 1.10, "High Risk": 1.45}
MODEL = {}


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS policyholders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_no TEXT UNIQUE NOT NULL,
        age INTEGER NOT NULL,
        vehicle_type TEXT NOT NULL,
        vehicle_age INTEGER NOT NULL,
        annual_km INTEGER NOT NULL,
        past_claims INTEGER NOT NULL,
        violations INTEGER NOT NULL,
        credit_score INTEGER NOT NULL,
        risk_score REAL,
        segment TEXT
    )""")
    conn.commit()
    conn.close()


def risk_score(r):
    """Rule-based risk score (0-100) used to rank the clusters."""
    score = (max(0, 25 - r["age"]) * 1.5 + max(0, r["age"] - 65) * 1.0
             + r["vehicle_age"] * 1.0 + r["annual_km"] / 1000 * 0.6
             + r["past_claims"] * 5 + r["violations"] * 4
             + (850 - r["credit_score"]) / 550 * 15)
    return round(min(100.0, score), 1)


def feature_row(r):
    """Driver age is converted to distance from the safest age (45)."""
    return [abs(float(r["age"]) - 45)] + [float(r[f]) for f in FEATURES[1:]]


def train_segments(rows, k=3):
    """K-Means clustering on scaled features; clusters ordered by mean risk score."""
    X = np.array([feature_row(r) for r in rows])
    scores = np.array([risk_score(r) for r in rows])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    order = np.argsort([scores[km.labels_ == c].mean() for c in range(k)])
    names = {int(c): SEGMENTS[i] for i, c in enumerate(order)}
    MODEL.update(scaler=scaler, km=km, names=names,
                 silhouette=round(float(silhouette_score(Xs, km.labels_)), 3))
    return [names[int(c)] for c in km.labels_], scores


def predict_segment(r):
    if not MODEL:
        return None
    x = MODEL["scaler"].transform([feature_row(r)])
    return MODEL["names"][int(MODEL["km"].predict(x)[0])]


def run_segmentation(conn):
    rows = conn.execute("SELECT * FROM policyholders").fetchall()
    if len(rows) < 3:
        return False
    labels, scores = train_segments(rows)
    for r, seg, sc in zip(rows, labels, scores):
        conn.execute("UPDATE policyholders SET segment=?, risk_score=? WHERE id=?",
                     (seg, float(sc), r["id"]))
    conn.commit()
    return True


def segment_summary(conn):
    return conn.execute("""SELECT segment, COUNT(*) AS customers,
        ROUND(AVG(age),1) AS age, ROUND(AVG(vehicle_age),1) AS vehicle_age,
        ROUND(AVG(annual_km)) AS annual_km, ROUND(AVG(past_claims),2) AS past_claims,
        ROUND(AVG(violations),2) AS violations, ROUND(AVG(credit_score)) AS credit_score,
        ROUND(AVG(risk_score),1) AS risk_score
        FROM policyholders WHERE segment IS NOT NULL GROUP BY segment
        ORDER BY AVG(risk_score)""").fetchall()


def logged_in():
    return session.get("logged_in")


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM policyholders").fetchone()[0]
    summary = segment_summary(conn)
    points = conn.execute("SELECT annual_km, risk_score, segment FROM policyholders "
                          "WHERE segment IS NOT NULL").fetchall()
    conn.close()
    counts = {s["segment"]: s["customers"] for s in summary}
    return render_template("dashboard.html", total=total, counts=counts,
                           summary=summary, points=points, segments=SEGMENTS)


@app.route("/policyholders", methods=["GET", "POST"])
def policyholders():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    if request.method == "POST":
        try:
            r = {f: int(request.form[f]) for f in FEATURES}
            seg = predict_segment(r)
            conn.execute(
                "INSERT INTO policyholders (policy_no, age, vehicle_type, vehicle_age, annual_km, "
                "past_claims, violations, credit_score, risk_score, segment) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request.form["policy_no"], r["age"], request.form["vehicle_type"],
                 r["vehicle_age"], r["annual_km"], r["past_claims"], r["violations"],
                 r["credit_score"], risk_score(r), seg))
            conn.commit()
            flash(f"Policyholder added - segment: {seg or 'run segmentation'}")
        except sqlite3.IntegrityError:
            flash("Policy number already exists")
        except ValueError:
            flash("Enter valid numeric values")
    rows = conn.execute("SELECT * FROM policyholders ORDER BY id DESC LIMIT 15").fetchall()
    conn.close()
    return render_template("policyholders.html", rows=rows)


@app.route("/segment")
def segment():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    if not run_segmentation(conn):
        conn.close()
        flash("Add at least three policyholders before segmentation")
        return redirect(url_for("policyholders"))
    summary = segment_summary(conn)
    conn.close()
    return render_template("segment.html", summary=summary,
                           silhouette=MODEL["silhouette"], loading=LOADING, base=BASE_PREMIUM)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        r = {f: int(request.form[f]) for f in FEATURES}
        seg = predict_segment(r)
        if seg:
            result = {"segment": seg, "score": risk_score(r),
                      "premium": round(BASE_PREMIUM * LOADING[seg])}
        else:
            flash("Run segmentation first")
    return render_template("predict.html", result=result, form=request.form)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    conn = get_db()
    run_segmentation(conn)
    conn.close()
    app.run(host="0.0.0.0", port=5000, debug=True)
