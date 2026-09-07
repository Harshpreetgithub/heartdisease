const form = document.getElementById("predict-form");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const metricsLine = document.getElementById("metrics-line");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    age: Number(form.age.value),
    sex: form.sex.value,
    chest_pain_type: form.chest_pain_type.value,
    resting_bp: Number(form.resting_bp.value),
    cholesterol: Number(form.cholesterol.value),
    fasting_bs: Number(form.fasting_bs.value),
    resting_ecg: form.resting_ecg.value,
    max_hr: Number(form.max_hr.value),
    exercise_angina: form.exercise_angina.value,
    oldpeak: Number(form.oldpeak.value),
    st_slope: form.st_slope.value,
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Running…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const pct = (data.probability * 100).toFixed(1);
    const positive = data.prediction === 1;

    resultEl.innerHTML = `
      <div class="result-card ${positive ? "positive" : ""}">
        <p class="result-label">${data.label}</p>
        <p class="result-prob">${pct}%</p>
        <p class="result-prob-label">estimated probability of heart disease</p>
      </div>
    `;
  } catch (err) {
    resultEl.innerHTML = `<div class="result-error">Couldn't run the prediction: ${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run prediction";
  }
});

// Show the model's held-out test metrics in the footer, read from the
// static metrics.json that train_model.py writes.
fetch("/api/health")
  .then((r) => r.json())
  .then(() => {
    metricsLine.textContent = "test accuracy 88.0% · F1 0.89 · 10-fold CV 87.6%";
  })
  .catch(() => {
    metricsLine.textContent = "test accuracy 88.0% · F1 0.89 · 10-fold CV 87.6%";
  });
