# Heart Disease Predictor

A heart disease prediction app I built using Random Forest. Takes 11 clinical inputs and gives a risk estimate.

## What it does

You fill in a form with patient details → the API predicts if they're at risk → shows a probability score.

## Results

Tested on 918 patients:
- Accuracy: 88.04%
- F1-score: 0.891
- Cross-validation: 87.60%

I compared 5 different models (Logistic Regression, KNN, SVM, Decision Tree, Random Forest) in my notebook and Random Forest was the best, so that's what I deployed.

## How it's built

**Frontend:** Static HTML form (index.html, styles.css, script.js)
**Backend:** FastAPI running on Vercel serverless
**Model:** Scikit-learn Random Forest with 100 trees

The pipeline:
1. Label encode sex and exercise angina
2. One-hot encode chest pain type, resting ECG, ST slope
3. Scale features with StandardScaler
4. Run through the trained Random Forest model

## Setup

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Train the model (optional - already trained models included)
python model/train_model.py

# Run locally
npm i -g vercel
vercel dev
```

Then go to http://localhost:3000

## Folder structure

```
├── api/
│   ├── index.py           # FastAPI backend
│   └── model/             # saved model files
├── model/
│   ├── train_model.py     # training script
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.json
├── data/
│   └── heart.csv
├── index.html
├── styles.css
├── script.js
└── requirements.txt
```

## Deploy

### Option 1: Vercel Dashboard
1. Push to GitHub
2. Go to vercel.com/new
3. Import repo, leave framework as "Other"
4. Deploy

### Option 2: Vercel CLI
```bash
vercel login
vercel --prod
```

## API

**POST /api/predict**

Request:
```json
{
  "age": 54,
  "sex": "M",
  "chest_pain_type": "ATA",
  "resting_bp": 130,
  "cholesterol": 246,
  "fasting_bs": 0,
  "resting_ecg": "Normal",
  "max_hr": 150,
  "exercise_angina": "N",
  "oldpeak": 1.0,
  "st_slope": "Up"
}
```

Response:
```json
{
  "prediction": 0,
  "label": "No Heart Disease Indicated",
  "probability": 0.12
}
```

**GET /api/health** → `{ "status": "ok", "model_loaded": true }`

## Things I should improve

- Add frontend validation so errors show before sending requests
- Write some pytest tests for the encoding logic
- Add rate limiting on the API before sharing it widely
- Keep a history of experiments instead of overwriting metrics

## Disclaimer

This is just a demo trained on public data from Kaggle. Not a real medical device. Don't use this instead of talking to a doctor.
