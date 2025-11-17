# Skill Obsolescence Predictor 

This project predicts the **future obsolescence of any tech skill** using a combination of **Google Trends**, **GitHub repository activity**, and **time-series forecasting (Exponential Smoothing)**.
It visualizes the historical trend, forecasts future interest for 3 years, and detects when a skill is expected to decline below a critical threshold.

---

## Features

✔ Fetches **weekly Google Trends data** (5 years)
✔ Fetches **GitHub repository counts** related to the skill
✔ Computes a **composite relevance score** (Trend + GitHub)
✔ Forecasts **3-year future interest** using
   • *Exponential Smoothing* (primary method)
   • *Linear Regression fallback*
✔ Detects **obsolescence date** using a 20% threshold for 12 consecutive weeks
✔ Generates a **trend + forecast plot** as Base64 image
✔ Provides a clean **Flask API** + simple HTML UI
✔ Includes **forecast evaluation metrics** (MAE, RMSE, MAPE)

---

## How It Works

### 1️⃣ Google Trends

Data is fetched weekly, resampled, and normalized (0–1 scale).
Caching is used to reduce API calls.

### 2️⃣ GitHub Popularity

Searches for repositories matching the skill in name, description, or README.
Counts are log-scaled into a 0–1 range.

### 3️⃣ Composite Score

```
composite = (0.7 * google_trends_normalized) + (0.3 * github_normalized)
```

### 4️⃣ Forecasting

Uses Holt-Winters Exponential Smoothing to project the next **156 weeks (3 years)**.

### 5️⃣ Obsolescence Detection

A skill is considered obsolete if:

> Forecast stays below **20% of peak popularity** for **12 consecutive weeks**

### 6️⃣ Output

Returns JSON with:

* Skill
* GitHub repo count
* Obsolescence prediction
* Base64 forecast plot

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
python app.py
```

The server starts at:

```
http://127.0.0.1:5000
```

---

## 📊 Forecast Plot

The system generates a plot containing:

* Historical composite trend
* 3-year forecast
* 20% obsolescence threshold
* Vertical line marking predicted obsolescence (if found)

Generated as Base64 → Embedded in UI.

---

## 📦 Caching & Rate Limits

* Google Trends requests are cached for **1 hour**.
* GitHub API may throttle without a token.
* Random sleep intervals help prevent Google blocking.
---

## 📜 License

MIT License

Just tell me!
