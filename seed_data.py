"""Generates a sample dataset of policyholders for demonstration."""
import random
import app

random.seed(18)
TYPES = ["Hatchback", "Sedan", "SUV", "Two-Wheeler"]


def make(i):
    profile = random.choices(["safe", "average", "risky"], [0.45, 0.35, 0.20])[0]
    if profile == "safe":
        age, va, km = random.randint(30, 60), random.randint(0, 5), random.randint(5000, 14000)
        claims, viol, credit = random.choice([0, 0, 0, 1]), random.choice([0, 0, 1]), random.randint(700, 850)
    elif profile == "average":
        age, va, km = random.randint(23, 68), random.randint(5, 10), random.randint(16000, 26000)
        claims, viol, credit = random.choice([1, 1, 2, 2]), random.choice([1, 2, 2, 3]), random.randint(560, 700)
    else:
        age, va, km = random.choice([random.randint(18, 24), random.randint(66, 75)]), random.randint(7, 15), random.randint(20000, 38000)
        claims, viol, credit = random.randint(3, 6), random.randint(3, 7), random.randint(300, 580)
    return (f"AIP{24000 + i}", age, random.choice(TYPES), va, km, claims, viol, credit)


if __name__ == "__main__":
    app.init_db()
    conn = app.get_db()
    conn.executemany(
        "INSERT OR IGNORE INTO policyholders (policy_no, age, vehicle_type, vehicle_age, annual_km, "
        "past_claims, violations, credit_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [make(i) for i in range(1, 501)])
    conn.commit()
    app.run_segmentation(conn)
    conn.close()
    print("Inserted sample policyholders and ran segmentation")
