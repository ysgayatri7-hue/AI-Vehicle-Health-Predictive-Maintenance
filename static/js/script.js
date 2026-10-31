// =========================================================
// AI VEHICLE HEALTH — CLIENT-SIDE UI BEHAVIOR
// No ML logic lives here. All predictions are computed by Flask.
// =========================================================

document.addEventListener("DOMContentLoaded", function () {
  initMobileNav();
  initScrollReveal();
  initRandomTestButton();
  initResultAnimations();
  initTimestamp();
  initShapWaterfall();
  initDashboardCharts();
});

// ---------------------------------------------------------
// Mobile navigation toggle
// ---------------------------------------------------------
function initMobileNav() {
  const toggle = document.querySelector(".nav-toggle");
  const panel = document.querySelector(".nav-mobile-panel");
  if (!toggle || !panel) return;

  toggle.addEventListener("click", function () {
    const isOpen = panel.classList.toggle("open");
    toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
  });

  panel.querySelectorAll("a").forEach(function (link) {
    link.addEventListener("click", function () {
      panel.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  });
}

// ---------------------------------------------------------
// Scroll-triggered fade-in reveal
// ---------------------------------------------------------
function initScrollReveal() {
  const items = document.querySelectorAll(".reveal");
  if (!items.length) return;

  if (!("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in-view"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  items.forEach((el) => observer.observe(el));
}

// ---------------------------------------------------------
// Random / preset test values for the prediction form
// Populates existing form fields only — no new endpoints.
// ---------------------------------------------------------
function initRandomTestButton() {
  const btn = document.getElementById("randomTestBtn");
  if (!btn) return;

  const scenarios = [
    // label, product_type, air_temp, process_temp, rpm, torque, tool_wear
    ["NORMAL", "M", 300, 310, 1500, 40, 100],
    ["HIGH TORQUE", "M", 300, 315, 1400, 70, 180],
    ["HIGH TOOL WEAR", "L", 302, 320, 1200, 60, 200],
  ];

  let scenarioIndex = 0;

  btn.addEventListener("click", function () {
    const useRandom = btn.dataset.mode === "random";
    let values;

    if (useRandom) {
      values = [
        "RANDOM",
        ["L", "M", "H"][Math.floor(Math.random() * 3)],
        randomInRange(295, 305),
        randomInRange(305, 322),
        randomInRange(1150, 1600),
        randomInRange(20, 75),
        randomInRange(0, 220),
      ];
    } else {
      values = scenarios[scenarioIndex % scenarios.length];
      scenarioIndex++;
    }

    const [, productType, airTemp, processTemp, rpm, torque, toolWear] = values;

    setFieldValue("product_type", productType);
    setFieldValue("air_temperature", airTemp);
    setFieldValue("process_temperature", processTemp);
    setFieldValue("rotational_speed", rpm);
    setFieldValue("torque", torque);
    setFieldValue("tool_wear", toolWear);

    flashPanel();
  });

  const randomToggle = document.getElementById("randomModeToggle");
  if (randomToggle) {
    randomToggle.addEventListener("click", function () {
      const isRandom = btn.dataset.mode === "random";
      btn.dataset.mode = isRandom ? "scenario" : "random";
      randomToggle.textContent = isRandom
        ? "Switch to fully random values"
        : "Switch to preset scenarios";
    });
  }
}

function randomInRange(min, max) {
  return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function setFieldValue(name, value) {
  const field = document.querySelector(`[name="${name}"]`);
  if (field) field.value = value;
}

function flashPanel() {
  const panel = document.querySelector(".analyzer-grid .panel");
  if (!panel) return;
  panel.style.transition = "border-color 0.2s ease";
  panel.style.borderColor = "rgba(244, 196, 0, 0.5)";
  setTimeout(() => {
    panel.style.borderColor = "";
  }, 400);
}

// ---------------------------------------------------------
// Result page: animate risk bar + health ring on load
// ---------------------------------------------------------
function initResultAnimations() {
  const riskFill = document.querySelector(".risk-bar-fill");
  if (riskFill) {
    const target = riskFill.dataset.value || "0";
    requestAnimationFrame(() => {
      setTimeout(() => {
        riskFill.style.width = target + "%";
      }, 150);
    });
  }

  const ringFill = document.querySelector(".health-ring .ring-fill");
  if (ringFill) {
    const value = parseFloat(ringFill.dataset.value || "0");
    const radius = parseFloat(ringFill.getAttribute("r"));
    const circumference = 2 * Math.PI * radius;
    ringFill.style.strokeDasharray = `${circumference}`;
    ringFill.style.strokeDashoffset = `${circumference}`;
    requestAnimationFrame(() => {
      setTimeout(() => {
        const offset = circumference - (value / 100) * circumference;
        ringFill.style.strokeDashoffset = `${offset}`;
      }, 150);
    });
  }
}

// ---------------------------------------------------------
// Non-authoritative generated-at timestamp for the report
// ---------------------------------------------------------
function initTimestamp() {
  const el = document.getElementById("reportTimestamp");
  if (!el) return;
  const now = new Date();
  const formatted = now.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  el.textContent = "Report generated " + formatted;
}

// ---------------------------------------------------------
// TreeSHAP explanation: waterfall sums to the checked XGBoost margin
// ---------------------------------------------------------
function initShapWaterfall() {
  const dataEl = document.getElementById("shap-data");
  const target = document.getElementById("shap-waterfall");
  if (!dataEl || !target || !window.Plotly) return;
  const explanation = JSON.parse(dataEl.textContent);
  const features = explanation.features || [];
  const x = ["Expected value", ...features.map((item) => `${item.feature} (${Number(item.value).toFixed(2)})`), "XGBoost margin"];
  const y = [Number(explanation.base), ...features.map((item) => Number(item.contribution)), 0];
  const measure = ["absolute", ...features.map(() => "relative"), "total"];
  const colors = { increasing: "#FF5C5C", decreasing: "#35D07F", totals: "#F4C400" };
  Plotly.newPlot(target, [{
    type: "waterfall", orientation: "v", measure, x, y, connector: { line: { color: "#647278" } },
    increasing: { marker: { color: colors.increasing } }, decreasing: { marker: { color: colors.decreasing } },
    totals: { marker: { color: colors.totals } },
    hovertemplate: "%{x}<br>Margin contribution: %{y:.4f}<extra></extra>",
  }], {
    paper_bgcolor: "#111F24", plot_bgcolor: "#111F24", font: { color: "#D7E3E8", family: "Inter, sans-serif" },
    margin: { t: 20, r: 24, b: 125, l: 65 }, xaxis: { tickangle: -35, automargin: true, gridcolor: "#24353B" },
    yaxis: { title: "Raw model margin (log-odds)", gridcolor: "#24353B", zerolinecolor: "#647278" },
    showlegend: false,
  }, { responsive: true, displaylogo: false });
}

// ---------------------------------------------------------
// Dashboard charts use persisted reports and AI4I SHAP values
// ---------------------------------------------------------
function initDashboardCharts() {
  if (!window.Plotly) return;
  const historyEl = document.getElementById("history-data");
  const historyTarget = document.getElementById("history-chart");
  if (historyEl && historyTarget) {
    const records = JSON.parse(historyEl.textContent);
    Plotly.newPlot(historyTarget, [{
      x: records.map((r) => r.created_at), y: records.map((r) => Number(r.failure_probability) * 100),
      customdata: records.map((r) => r.id), type: "scatter", mode: "lines+markers",
      line: { color: "#F4C400", width: 2 }, marker: { color: "#F4C400", size: 7 },
      hovertemplate: "Report #%{customdata}<br>%{x}<br>Failure probability: %{y:.2f}%<extra></extra>",
    }], chartLayout("Failure probability (%)"), { responsive: true, displaylogo: false });
  }
  const importanceEl = document.getElementById("importance-data");
  const importanceTarget = document.getElementById("importance-chart");
  if (importanceEl && importanceTarget) {
    const items = JSON.parse(importanceEl.textContent).slice().reverse();
    Plotly.newPlot(importanceTarget, [{
      x: items.map((x) => x.mean_abs_shap), y: items.map((x) => x.feature),
      type: "bar", orientation: "h", marker: { color: "#F4C400" },
      hovertemplate: "%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    }], { ...chartLayout("Mean absolute SHAP value"), margin: { t: 12, r: 24, b: 55, l: 190 } }, { responsive: true, displaylogo: false });
  }
}

function chartLayout(xTitle) {
  return {
    paper_bgcolor: "#111F24", plot_bgcolor: "#111F24", font: { color: "#D7E3E8", family: "Inter, sans-serif" },
    margin: { t: 15, r: 24, b: 65, l: 65 }, xaxis: { title: xTitle, gridcolor: "#24353B" },
    yaxis: { gridcolor: "#24353B", automargin: true },
  };
}

