# Auto Insurance Risk Segmentation

DevOps & Agile Methodologies (23AD52C) mini project - ABHINAV S P (24243018), AI & DS, National Engineering College.

A Flask web app that groups auto-insurance policyholders into Low, Medium and High risk segments using
K-Means clustering on driver and vehicle features, profiles each segment, suggests premiums and predicts
the segment of a new customer.

## Run
    pip install -r requirements.txt
    python seed_data.py      # loads 500 sample policyholders
    python app.py            # http://localhost:5000  (admin / admin123)

## Test
    pytest -v

## Docker
    docker compose up --build
