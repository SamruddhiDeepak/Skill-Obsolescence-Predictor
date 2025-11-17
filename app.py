import os
import time
import random
import math
from datetime import datetime, timedelta
from io import BytesIO
import base64

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


try:
    from pytrends.request import TrendReq
except ImportError:
    raise SystemExit("pytrends required. Install with: pip install pytrends")

# CONFIG
PYTRENDS_TIMEFRAME = "today 5-y"
FORECAST_WEEKS = 52*3
OBSOL_FRAC = 0.20
CONSEC_WEEKS = 12
USE_GITHUB = True
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)

WEIGHT_TRENDS = 0.7
WEIGHT_GITHUB = 0.3
GITHUB_CAP = 100000
TRENDS_CACHE = {}
CACHE_EXPIRY = 60 * 60   

# HELPERS
def safe_sleep(a=1.0, b=2.0):
    time.sleep(random.uniform(a, b))

def fetch_google_trends_weekly(skill, timeframe=PYTRENDS_TIMEFRAME):
    now = time.time()

    # --- 1) Return from CACHE if fresh ---
    if skill in TRENDS_CACHE:
        cached_time, cached_data = TRENDS_CACHE[skill]
        if now - cached_time < CACHE_EXPIRY:
            print("Serving from CACHE:", skill)
            return cached_data

    # --- 2) Rotate user agents ---
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPad; CPU OS 13_2 like Mac OS X)",
    ]

    attempts = 0
    while attempts < 6:
        try:
            pytrends = TrendReq(
                hl='en-US',
                tz=330,
                requests_args={'headers': {'User-Agent': random.choice(USER_AGENTS)}}
            )

            safe_sleep(1.5, 3.0)  # important slow-down

            pytrends.build_payload([skill], cat=0, timeframe=timeframe, geo='', gprop='')
            df = pytrends.interest_over_time()

            # validation
            if df is None or df.empty:
                raise Exception("Empty trends data")

            series = df[skill]
            if "isPartial" in df:
                series = series[~df["isPartial"]]

            # weekly interpolation
            series = series.resample("W-MON").mean().interpolate()

            # ---- store in CACHE ----
            TRENDS_CACHE[skill] = (now, series)

            return series

        except Exception as e:
            attempts += 1
            wait = random.uniform(4, 10) * attempts
            print(f"Google Trends ERROR {attempts}: {e}  -> retrying in {wait:.1f}s")
            time.sleep(wait)

    return None

def fetch_github_repo_count(skill):
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "skill-weekly-script"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    query = requests.utils.quote(f"{skill} in:name,description,readme")
    url = f"https://api.github.com/search/repositories?q={query}&per_page=1"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            j = r.json()
            return int(j.get("total_count", 0))
        else:
            return None
    except:
        return None

def log_scale_to_0_1(x, cap=GITHUB_CAP):
    if x is None or x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log1p(cap))

# CORE
def forecast_for_frontend(skill):
    trends_series = fetch_google_trends_weekly(skill)
    if trends_series is None:
        return {"error": "Google Trends fetch failed"}

    # normalize trends
    trends_norm = (trends_series - trends_series.min()) / (trends_series.max() - trends_series.min() + 1e-9)
    weekly_df = pd.DataFrame({"trend": trends_norm}).rename_axis("date").reset_index()
    weekly_df.set_index("date", inplace=True)

    # GitHub normalization
    github_norm = 0.0
    github_count = None
    if USE_GITHUB:
        github_count = fetch_github_repo_count(skill)
        github_norm = log_scale_to_0_1(github_count)

    # composite
    weekly_df["composite"] = weekly_df["trend"] * WEIGHT_TRENDS + github_norm * WEIGHT_GITHUB

    # forecast
    series = weekly_df["composite"].values
    try:
        model = ExponentialSmoothing(series, trend="add")
        fit = model.fit(optimized=True)
        fc = fit.forecast(FORECAST_WEEKS)
    except:
        x = np.arange(len(series))
        coef = np.polyfit(x, series, 1)
        fc = np.polyval(coef, np.arange(len(series), len(series)+FORECAST_WEEKS))
    fc = np.clip(fc, 0, 1)

    last_date = weekly_df.index[-1]
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(FORECAST_WEEKS)]
    forecast_df = pd.DataFrame({"forecast": fc}, index=future_dates)

    # CALCULATE FORECAST ERRORS
    n_eval = min(len(series), len(fc))
    actual = series[-n_eval:]
    predicted = fc[:n_eval]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100  # percentage

    print(f"=== Forecast Evaluation Metrics for '{skill}' ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("==============================================")

    # obsolescence detection
    peak = weekly_df["composite"].max()
    threshold = peak * OBSOL_FRAC
    below = (forecast_df["forecast"] < threshold).astype(int).values
    consec = 0
    obsol_date = None
    for i, b in enumerate(below):
        if b:
            consec += 1
            if consec >= CONSEC_WEEKS:
                obsol_date = forecast_df.index[i - CONSEC_WEEKS + 1]
                break
        else:
            consec = 0

    # plot to base64
    plt.figure(figsize=(10,4))
    plt.plot(weekly_df.index, weekly_df["composite"], label="Composite (history)")
    plt.plot(forecast_df.index, forecast_df["forecast"], linestyle="--", label="Forecast")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({OBSOL_FRAC*100:.0f}% peak)")
    if obsol_date:
        plt.axvline(obsol_date, color="red", linestyle=":", label=f"Obsolescence ~{obsol_date.date()}")
    plt.legend()
    plt.grid(alpha=0.3)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "skill": skill,
        "history_len": len(weekly_df),
        "peak": float(peak),
        "threshold": float(threshold),
        "github_count": github_count,
        "obsolescence_msg": "No obsolescence" if not obsol_date else f"Obsolescence ~{obsol_date.date()}",
        "plot_base64": plot_base64
    }

# FLASK APP
app = Flask(__name__)

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Skill Obsolescence Predictor</title></head>
    <body>
    <h1>Skill Obsolescence Predictor</h1>
    <form id="skillForm">
        Skill name: <input type="text" id="skill" name="skill">
        <button type="submit">Predict</button>
    </form>
    <div id="result"></div>
    <script>
    const form = document.getElementById('skillForm');
    form.onsubmit = async (e) => {
        e.preventDefault();
        const skill = document.getElementById('skill').value;
        const resDiv = document.getElementById('result');
        resDiv.innerHTML = "Loading...";
        try {
            const resp = await fetch(`/predict?skill=${skill}`);
            const data = await resp.json();
            if (data.error) { resDiv.innerHTML = data.error; return; }
            resDiv.innerHTML = `
                <p>Skill: ${data.skill}</p>
                <p>GitHub repo count: ${data.github_count}</p>
                <p>${data.obsolescence_msg}</p>
                <img src="data:image/png;base64,${data.plot_base64}" style="max-width:100%">
            `;
        } catch(err){ resDiv.innerHTML = "Error: "+err; }
    };
    </script>
    </body>
    </html>
    """

@app.route("/predict")
def predict():
    skill = request.args.get("skill")
    if not skill:
        return jsonify({"error": "Skill parameter required"})
    return jsonify(forecast_for_frontend(skill))

if __name__ == "__main__":
    app.run(debug=True)
