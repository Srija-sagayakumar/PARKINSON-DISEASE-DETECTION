function scrollToSection(id) {
  document.getElementById(id).scrollIntoView({ behavior: "smooth" });
}

const API_BASE_URL = window.location.origin;
const DEFAULT_SAMPLING_RATE = 100;

particlesJS("particles-js", {
  particles: {
    number: { value: 80 },
    size: { value: 2 },
    color: { value: "#00f0ff" },
    line_linked: { enable: true, color: "#00f0ff" },
    move: { speed: 2 }
  }
});

function computeFFT(signal) {
  const N = signal.length;
  const out = [];
  for (let k = 0; k < N; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im -= signal[n] * Math.sin(angle);
    }
    out.push(Math.sqrt(re * re + im * im));
  }
  return out;
}

function plot(id, traces, title) {
  Plotly.newPlot(
    id,
    traces,
    {
      title: { text: title || "", font: { color: "#d8f6ff", size: 14 } },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "white" },
      margin: { l: 40, r: 20, t: 40, b: 40 },
      xaxis: { gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { gridcolor: "rgba(255,255,255,0.08)" }
    },
    { responsive: true }
  );
}

function setStatus(statusClass, statusText) {
  const indicator = document.getElementById("statusIndicator");
  indicator.className = "status-indicator " + statusClass;
  document.getElementById("statusText").innerText = statusText;
}

function renderGauge(parkinsonProbability) {
  const value = Number(parkinsonProbability || 0);
  Plotly.newPlot(
    "gaugeChart",
    [{
      type: "indicator",
      mode: "gauge+number",
      value,
      number: { suffix: "%", font: { color: "#ffffff", size: 34 } },
      title: { text: "Parkinson's Disease Probability", font: { color: "#d8f6ff", size: 14 } },
      gauge: {
        axis: { range: [0, 100], tickcolor: "#fff" },
        bar: { color: "#00e5ff" },
        bgcolor: "rgba(0,0,0,0)",
        borderwidth: 1,
        bordercolor: "rgba(255,255,255,0.2)",
        steps: [
          { range: [0, 33], color: "rgba(0, 255, 156, 0.35)" },
          { range: [33, 66], color: "rgba(255, 193, 7, 0.35)" },
          { range: [66, 100], color: "rgba(255, 77, 77, 0.4)" }
        ],
        threshold: {
          line: { color: "#ffffff", width: 3 },
          thickness: 0.75,
          value
        }
      }
    }],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      margin: { l: 10, r: 10, t: 50, b: 10 }
    },
    { responsive: true }
  );
}

function parseCsvRows(text) {
  const rows = text.split(/\r?\n/).filter(Boolean);
  if (rows.length < 2) return null;

  const headers = rows[0].split(",").map((h) => h.trim());
  const required = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"];
  const indices = Object.fromEntries(required.map((key) => [key, headers.indexOf(key)]));

  if (required.some((key) => indices[key] === -1)) {
    return null;
  }

  const parsed = {
    xAcc: [], yAcc: [], zAcc: [],
    xGyro: [], yGyro: [], zGyro: []
  };

  for (let i = 1; i < rows.length; i++) {
    const cols = rows[i].split(",");
    const row = {
      xAcc: Number(cols[indices.xAcc]),
      yAcc: Number(cols[indices.yAcc]),
      zAcc: Number(cols[indices.zAcc]),
      xGyro: Number(cols[indices.xGyro]),
      yGyro: Number(cols[indices.yGyro]),
      zGyro: Number(cols[indices.zGyro])
    };
    if (Object.values(row).some((v) => Number.isNaN(v))) continue;
    parsed.xAcc.push(row.xAcc);
    parsed.yAcc.push(row.yAcc);
    parsed.zAcc.push(row.zAcc);
    parsed.xGyro.push(row.xGyro);
    parsed.yGyro.push(row.yGyro);
    parsed.zGyro.push(row.zGyro);
  }

  return parsed.xAcc.length ? parsed : null;
}

function renderWaveform(parsed) {
  const limit = Math.min(parsed.xAcc.length, 600);
  const x = Array.from({ length: limit }, (_, i) => i);

  const traces = [
    { x, y: parsed.xAcc.slice(0, limit), type: "scatter", mode: "lines", name: "xAcc", line: { color: "#00e5ff", width: 2 } },
    { x, y: parsed.yAcc.slice(0, limit), type: "scatter", mode: "lines", name: "yAcc", line: { color: "#7CFF6B", width: 1.5 } },
    { x, y: parsed.zAcc.slice(0, limit), type: "scatter", mode: "lines", name: "zAcc", line: { color: "#ff9f43", width: 1.5 } }
  ];

  plot("signalChart", traces, "Accelerometer Waveform (first 600 samples)");
}

function renderTremorSpectrum(parsed, samplingRate = DEFAULT_SAMPLING_RATE) {
  const n = Math.min(parsed.xAcc.length, 1024);
  if (n < 32) return;

  const accMag = [];
  for (let i = 0; i < n; i++) {
    const m = Math.sqrt(
      parsed.xAcc[i] * parsed.xAcc[i] +
      parsed.yAcc[i] * parsed.yAcc[i] +
      parsed.zAcc[i] * parsed.zAcc[i]
    );
    accMag.push(m);
  }

  const mean = accMag.reduce((a, b) => a + b, 0) / accMag.length;
  const centered = accMag.map((v) => v - mean);

  const amps = computeFFT(centered);
  const half = Math.floor(n / 2);
  const freq = [];
  const mag = [];
  for (let k = 0; k <= half; k++) {
    freq.push((k * samplingRate) / n);
    mag.push(amps[k] / n);
  }

  const maxFreq = 15;
  const filteredFreq = [];
  const filteredMag = [];
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] <= maxFreq) {
      filteredFreq.push(freq[i]);
      filteredMag.push(mag[i]);
    }
  }

  Plotly.newPlot(
    "fftChart",
    [{
      x: filteredFreq,
      y: filteredMag,
      type: "scatter",
      mode: "lines",
      line: { color: "#7bdff2", width: 2 },
      name: "Spectrum"
    }],
    {
      title: { text: "Tremor Frequency Spectrum (0-15 Hz)", font: { color: "#d8f6ff", size: 14 } },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "white" },
      margin: { l: 45, r: 20, t: 40, b: 40 },
      xaxis: { title: "Frequency (Hz)", gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { title: "Magnitude", gridcolor: "rgba(255,255,255,0.08)" },
      shapes: [
        {
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: 4,
          x1: 6,
          y0: 0,
          y1: 1,
          fillcolor: "rgba(255, 193, 7, 0.18)",
          line: { width: 0 }
        }
      ],
      annotations: [
        {
          x: 5,
          y: 1,
          yref: "paper",
          text: "Parkinson tremor band (4-6 Hz)",
          showarrow: false,
          font: { color: "#ffd166", size: 11 }
        }
      ]
    },
    { responsive: true }
  );
}

function setResultCard(data) {
  const predictionEl = document.getElementById("prediction");
  const diagnosisEl = document.getElementById("diagnosis");
  const riskEl = document.getElementById("risk");
  const finalResultEl = document.getElementById("finalResult");

  predictionEl.innerText = `Predicted Class: ${data.predicted_label}`;
  diagnosisEl.innerText = data.diagnosis_text;

  const normal = Number(data.condition_percentages?.Normal || 0).toFixed(2);
  const mild = Number(data.condition_percentages?.["Mild Tremor"] || 0).toFixed(2);
  const parkinson = Number(data.condition_percentages?.["Parkinson's Disease"] || 0).toFixed(2);

  riskEl.innerText = `Normal: ${normal}% | Mild Tremor: ${mild}% | Parkinson's Disease: ${parkinson}%`;
  renderGauge(parkinson);

  if (data.parkinsons_detected === true) {
    finalResultEl.innerText = "YES";
    setStatus("status-critical", "Parkinson's Pattern Detected");
  } else if (data.parkinsons_detected === false) {
    finalResultEl.innerText = "NO";
    setStatus("status-safe", "No Parkinson's Pattern Detected");
  } else {
    finalResultEl.innerText = "INCONCLUSIVE";
    setStatus("status-risk", "Inconclusive / Mixed Pattern");
  }
}

async function predict() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  const loader = document.getElementById("loader");

  if (!file) {
    alert("Upload a CSV file first.");
    return;
  }

  loader.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    setResultCard(payload);
  } catch (error) {
    document.getElementById("prediction").innerText = "Prediction failed";
    document.getElementById("diagnosis").innerText = String(error.message || error);
    document.getElementById("risk").innerText = "";
    document.getElementById("finalResult").innerText = "---";
    setStatus("status-risk", "Error");
  } finally {
    loader.classList.add("hidden");
  }
}

document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (ev) {
    const parsed = parseCsvRows(String(ev.target.result || ""));
    if (!parsed) return;

    renderWaveform(parsed);
    renderTremorSpectrum(parsed);
  };
  reader.readAsText(file);
});

function simulate() {
  const f = Number(freq.value);
  const a = Number(amp.value);
  const n = Number(noise.value);
  const x = [];
  const y = [];
  for (let i = 0; i < 256; i++) {
    x.push(i);
    y.push(a * Math.sin(i / f) + Math.random() * n);
  }
  plot("simChart", [{ x, y, type: "scatter", mode: "lines", line: { color: "#7bdff2" } }], "Simulation Waveform");
}

Plotly.newPlot(
  "dataChart",
  [{ values: [70, 20, 10], labels: ["Normal", "Tremor", "Parkinson"], type: "pie" }],
  { paper_bgcolor: "transparent", font: { color: "white" } }
);

document.querySelectorAll(".faq-question").forEach((btn) => {
  btn.addEventListener("click", () => {
    const item = btn.parentElement;
    document.querySelectorAll(".faq-item").forEach((i) => {
      if (i !== item) i.classList.remove("active");
    });
    item.classList.toggle("active");
  });
});

function downloadReport() {
  const finalResult = document.getElementById("finalResult").innerText;
  const prediction = document.getElementById("prediction").innerText;
  const diagnosis = document.getElementById("diagnosis").innerText;
  const risk = document.getElementById("risk").innerText;

  const text = `NeuroDetect AI Report\n\nResult: ${finalResult}\n${prediction}\n${diagnosis}\n${risk}`;
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "report.txt";
  a.click();
}

const canvas = document.getElementById("bgCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let t = 0;
function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  for (let x = 0; x < canvas.width; x++) {
    const y = canvas.height / 2 + Math.sin(x * 0.01 + t) * 50;
    ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "cyan";
  ctx.stroke();
  t += 0.05;
  requestAnimationFrame(animate);
}
animate();

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.innerText = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2500);
}

document.getElementById("sendBtn")?.addEventListener("click", () => {
  const name = document.getElementById("contactName").value.trim();
  const email = document.getElementById("contactEmail").value.trim();
  const message = document.getElementById("contactMessage").value.trim();

  if (!name || !email || !message) {
    showToast("Please fill all contact details.");
    return;
  }

  showToast("Thanks! Your message has been received.");
  document.getElementById("contactName").value = "";
  document.getElementById("contactEmail").value = "";
  document.getElementById("contactMessage").value = "";
});
