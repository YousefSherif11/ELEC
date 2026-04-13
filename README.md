# Electronics Pricing Streamlit App

A professional Streamlit dashboard for analyzing the Datafiniti electronics pricing dataset and predicting average product prices.

## Files
- `streamlitapp.py` — Streamlit dashboard and price predictor
- `DatafinitiElectronicsProductsPricingData.csv` — dataset file (ignored in Git by default)
- `model.pkl` — saved machine learning pipeline for predictions
- `requirements.txt` — Python dependencies

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run locally
```bash
streamlit run streamlitapp.py
```

## Deploy to Streamlit Cloud
1. Push this repository to GitHub.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**.
4. Select your repository, branch `main`, and file `streamlitapp.py`.
5. Click **Deploy**.

Once deployed, the public app URL will look like:

`https://share.streamlit.io/essraaadel/electronicsprices/main/streamlitapp.py`

> Replace `essraaadel` and `electronicsprices` with your GitHub details if you rename the repository.

[![Open in Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/essraaadel/electronicsprices/main/streamlitapp.py)

## Notes
- If `DatafinitiElectronicsProductsPricingData.csv` is present, the app loads real dataset statistics.
- If `model.pkl` is present, the Price Predictor page becomes active.
- If either file is missing, the app falls back to representative sample data and displays guidance.
