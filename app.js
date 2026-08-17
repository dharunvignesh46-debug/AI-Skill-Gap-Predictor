const API_BASE = "http://localhost:8000";

const SKILL_DEFAULTS = {
  Python: 55, Machine_Learning: 50, SQL: 55, Deep_Learning: 40,
  NLP: 35, DSA: 45, Git: 50, Cloud: 40, Statistics: 50, Communication: 55,
};

let radarChart = null;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  buildMeters();
  wireStaticEvents();

  try {
    const [health, roles] = await Promise.all([
      fetchJSON("/api/health"),
      fetchJSON("/api/roles"),
    ]);
    setApiOnline(health.metrics);
    populateRoles(roles.roles);
  } catch (err) {
    setApiOffline();
    console.error(err);
  }
}

function fetchJSON(path, options) {
  return fetch(API_BASE + path, options).then((r) => {
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return r.json();
  });
}

function setApiOnline(metrics) {
  document.getElementById("apiStatusDot").classList.add("online");
  document.getElementById("apiStatusText").textContent = "api online";
  document.getElementById("modelMeta").textContent =
    `classifier acc ${(metrics.classifier_accuracy * 100).toFixed(1)}%  ·  regressor R² ${metrics.regressor_r2.toFixed(2)}`;
}

function setApiOffline() {
  document.getElementById("apiStatusText").textContent = "api offline — start backend on :8000";
}

function populateRoles(roles) {
  const sel = document.getElementById("targetRole");
  sel.innerHTML = roles.map((r) => `<option value="${r}">${r}</option>`).join("");
}

// ---------------------------------------------------------------------------
// Skill meters
// ---------------------------------------------------------------------------
function buildMeters() {
  const container = document.getElementById("meters");
  container.innerHTML = Object.entries(SKILL_DEFAULTS)
    .map(
      ([skill, val]) => `
      <div class="meter" data-skill="${skill}">
        <div class="meter-head">
          <span class="meter-name">${skill.replace(/_/g, " ")}</span>
          <span class="meter-val mono">${val}</span>
        </div>
        <div class="meter-track">
          <div class="meter-fill" style="width:${val}%"></div>
          <input type="range" min="0" max="100" value="${val}" />
        </div>
      </div>`
    )
    .join("");

  container.querySelectorAll(".meter").forEach((meterEl) => {
    const input = meterEl.querySelector("input[type=range]");
    const fill = meterEl.querySelector(".meter-fill");
    const valLabel = meterEl.querySelector(".meter-val");
    input.addEventListener("input", () => {
      fill.style.width = input.value + "%";
      valLabel.textContent = input.value;
    });
  });
}

function readSkillValues() {
  const values = {};
  document.querySelectorAll(".meter").forEach((meterEl) => {
    const skill = meterEl.dataset.skill;
    values[skill] = Number(meterEl.querySelector("input[type=range]").value);
  });
  return values;
}

// ---------------------------------------------------------------------------
// Scan action
// ---------------------------------------------------------------------------
function wireStaticEvents() {
  document.getElementById("scanBtn").addEventListener("click", runScan);
}

async function runScan() {
  const btn = document.getElementById("scanBtn");
  btn.disabled = true;
  btn.querySelector(".scan-btn-label").textContent = "Scanning…";

  const payload = {
    ...readSkillValues(),
    Projects: Number(document.getElementById("projects").value),
    Certifications: Number(document.getElementById("certifications").value),
    Years_Experience: Number(document.getElementById("experience").value),
    Internships: Number(document.getElementById("internships").value),
    Target_Role: document.getElementById("targetRole").value,
  };

  try {
    const result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResult(result);
  } catch (err) {
    alert("Error: " + err.message);
    console.error("FULL ERROR:", err);
  } finally {
    btn.disabled = false;
    btn.querySelector(".scan-btn-label").textContent = "Run scan";
  }
}

// ---------------------------------------------------------------------------
// Render results
// ---------------------------------------------------------------------------
function renderResult(result) {
  document.getElementById("readoutEmpty").hidden = true;
  const content = document.getElementById("readoutContent");
  content.hidden = false;

  renderGauge(result.job_readiness_score);
  renderLevel(result.skill_gap_level, result.skill_gap_level_confidence);
  document.getElementById("strongestValue").textContent = result.strongest_skill;
  renderRadar(result.skill_breakdown);
  renderRecommendations(result.recommendations);
}

function renderGauge(score) {
  const circumference = 251.2; // matches stroke-dasharray in CSS (approx semi-circle path length)
  const offset = circumference * (1 - score / 100);
  const fill = document.getElementById("gaugeFill");
  fill.style.strokeDashoffset = offset;

  let color = "var(--coral)";
  if (score >= 75) color = "var(--green)";
  else if (score >= 50) color = "var(--cyan)";
  else if (score >= 30) color = "var(--amber)";
  fill.style.stroke = color;

  document.getElementById("gaugeValue").textContent = Math.round(score);
}

function renderLevel(level, confidence) {
  const badge = document.getElementById("levelBadge");
  badge.textContent = level.toUpperCase();
  badge.className = "level-badge " + level;

  const confText = Object.entries(confidence)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
    .join("  ·  ");
  document.getElementById("levelConfidence").textContent = confText;
}

function renderRadar(breakdown) {
  const labels = breakdown.map((s) => s.skill.replace(/_/g, " "));
  const userData = breakdown.map((s) => s.user_score);
  const benchmarkData = breakdown.map((s) => s.role_benchmark);

  const ctx = document.getElementById("radarChart").getContext("2d");

  if (radarChart) {
    radarChart.data.labels = labels;
    radarChart.data.datasets[0].data = userData;
    radarChart.data.datasets[1].data = benchmarkData;
    radarChart.update();
    return;
  }

  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "You",
          data: userData,
          borderColor: "#52D9CB",
          backgroundColor: "rgba(82,217,203,0.15)",
          pointBackgroundColor: "#52D9CB",
          borderWidth: 2,
        },
        {
          label: "Role average",
          data: benchmarkData,
          borderColor: "#4E6670",
          backgroundColor: "rgba(78,102,112,0.08)",
          pointBackgroundColor: "#4E6670",
          borderWidth: 1.5,
          borderDash: [4, 3],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0,
          max: 100,
          angleLines: { color: "#28394370" },
          grid: { color: "#28394370" },
          pointLabels: { color: "#83A0A9", font: { size: 11, family: "IBM Plex Mono" } },
          ticks: { display: false, backdropColor: "transparent" },
        },
      },
    },
  });
}

function renderRecommendations(recs) {
  const list = document.getElementById("recList");
  if (!recs.length) {
    list.innerHTML = `<li class="none">No priority gaps — you're at or above benchmark on every skill.</li>`;
    return;
  }
  list.innerHTML = recs
    .map(
      (r, i) => `
      <li>
        <span class="rec-rank mono">0${i + 1}</span>
        <div class="rec-body">
          <div class="rec-skill">${r.skill}</div>
          <div class="rec-msg">${r.message}</div>
        </div>
        <span class="rec-gap mono">+${r.gap}</span>
      </li>`
    )
    .join("");
}

init();
