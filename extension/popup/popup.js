/**
 * Popup Script for Personalized CVD Assistant.
 * Handles profile configuration, health monitoring, and interactive screening.
 */

const API_BASE = "http://localhost:8000";
const $ = id => document.getElementById(id);

let plates = [];
let answers = [];
let busy = false;
let latestDiagnosis = null;

async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const res = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json", ...options.headers },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `Server error: ${res.status}`);
    }
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  }
}

async function checkHealth() {
  try {
    const data = await fetchApi("/health");
    if (data.trained_weights_loaded) {
      $("apiIndicator").className = "indicator online";
      $("apiStatusText").textContent = `${data.architecture} ready · ${data.device}`;
      return true;
    } else {
      $("apiIndicator").className = "indicator degraded";
      $("apiStatusText").textContent = "Backend ready (random weights active)";
      return true;
    }
  } catch (err) {
    $("apiIndicator").className = "indicator offline";
    $("apiStatusText").textContent = "Backend offline (start python run_server.py)";
    return false;
  }
}

async function loadSettings() {
  const state = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope", "auto_diagnosed"]);
  $("enableToggle").checked = state.enabled ?? false;
  $("toggleStatusLabel").textContent = state.enabled ? "ON" : "OFF";
  $("conditionSelect").value = String(state.condition_id !== undefined ? state.condition_id : 1);
  const sevPercent = Math.round((state.severity !== undefined ? state.severity : 0.85) * 100);
  $("severitySlider").value = sevPercent;
  $("severityValue").textContent = `${sevPercent}%`;

  const scope = state.scope || "full_page";
  if (scope === "images_only") {
    $("scopeImagesOnly").checked = true;
  } else {
    $("scopeFullPage").checked = true;
  }

  if (state.auto_diagnosed) {
    $("detectedBadge").classList.remove("hidden");
  } else {
    $("detectedBadge").classList.add("hidden");
  }
}

async function saveSettings(autoDiagnosed = false) {
  const enabled = $("enableToggle").checked;
  const conditionId = Number($("conditionSelect").value);
  const severity = Number($("severitySlider").value) / 100.0;
  const scope = $("scopeImagesOnly").checked ? "images_only" : "full_page";

  await chrome.storage.local.set({
    enabled: enabled,
    condition_id: conditionId,
    severity: severity,
    scope: scope,
    auto_diagnosed: autoDiagnosed,
  });

  $("toggleStatusLabel").textContent = enabled ? "ON" : "OFF";

  // Notify active tab
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      await chrome.tabs.sendMessage(tab.id, { action: "SETTINGS_UPDATED" });
    }
  } catch (_) {
    // Content script might not be injected on special browser pages
  }
}

function switchTab(targetName) {
  const isSettings = targetName === "Dashboard";
  $("tabDashboard").classList.toggle("active", isSettings);
  $("tabDiagnosis").classList.toggle("active", !isSettings);
  $("viewDashboard").classList.toggle("active-view", isSettings);
  $("viewDiagnosis").classList.toggle("active-view", !isSettings);
}

async function startScreeningTest() {
  answers = [];
  latestDiagnosis = null;
  $("testResultWrapper").classList.add("hidden");
  $("testContainer").classList.remove("hidden");

  try {
    const data = await fetchApi("/api/v1/plates");
    plates = data.plates || [];
    if (!data.available || plates.length === 0) {
      throw new Error(data.message || "No screening plates installed.");
    }
    $("screeningNotice").textContent = data.message;
    renderCurrentPlate();
  } catch (err) {
    $("testContainer").classList.add("hidden");
    $("screeningNotice").textContent = `Screening unavailable: ${err.message}`;
  }
}

function renderCurrentPlate() {
  busy = false;
  const currentIndex = answers.length;
  const plate = plates[currentIndex];

  $("testProgress").style.width = `${(currentIndex / plates.length) * 100}%`;
  $("plateQuestion").textContent = `Plate ${currentIndex + 1} of ${plates.length}`;
  $("platePrompt").textContent = plate.prompt || "What numeral do you perceive in the plate?";

  // Plate image
  const img = document.createElement("img");
  img.alt = "Screening plate graphic";
  img.src = plate.image;
  $("plateGraphic").replaceChildren(img);

  // Options buttons
  $("plateOptions").replaceChildren();
  for (const option of plate.options) {
    const btn = document.createElement("button");
    btn.className = "option-btn";
    btn.textContent = option;
    btn.onclick = async () => {
      if (busy) return;
      busy = true;

      answers.push({ plate_id: plate.id, user_answer: option });

      if (answers.length < plates.length) {
        renderCurrentPlate();
      } else {
        await finishScreening();
      }
    };
    $("plateOptions").append(btn);
  }
}

async function finishScreening() {
  $("testContainer").classList.add("hidden");
  $("plateOptions").replaceChildren();

  try {
    const result = await fetchApi("/api/v1/diagnose", {
      method: "POST",
      body: JSON.stringify({ answers }),
    });

    latestDiagnosis = result;
    $("diagnosisTitle").textContent = result.condition;
    $("diagnosisDetail").textContent = `${result.errors} of ${result.total} screening errors. ${result.message}`;
    $("testResultWrapper").classList.remove("hidden");
  } catch (err) {
    $("diagnosisTitle").textContent = "Evaluation Failed";
    $("diagnosisDetail").textContent = err.message;
    $("testResultWrapper").classList.remove("hidden");
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  // Navigation
  $("tabDashboard").onclick = () => switchTab("Dashboard");
  $("tabDiagnosis").onclick = () => {
    switchTab("Diagnosis");
    startScreeningTest();
  };

  // Settings controls
  $("enableToggle").onchange = async () => {
    if ($("enableToggle").checked) {
      const isOnline = await checkHealth();
      if (!isOnline) {
        // Still allow enabling with warning
      }
    }
    await saveSettings();
  };

  $("conditionSelect").onchange = () => {
    $("detectedBadge").classList.add("hidden");
    saveSettings(false);
  };

  $("scopeFullPage").onchange = () => saveSettings(false);
  $("scopeImagesOnly").onchange = () => saveSettings(false);

  $("severitySlider").oninput = () => {
    $("severityValue").textContent = `${$("severitySlider").value}%`;
  };
  $("severitySlider").onchange = () => saveSettings(false);

  $("applyNowBtn").onclick = async () => {
    await saveSettings();
    $("apiStatusText").textContent = "Settings saved & applied.";
  };

  $("resetProfileBtn").onclick = async () => {
    $("enableToggle").checked = false;
    $("conditionSelect").value = "1";
    $("severitySlider").value = "85";
    $("severityValue").textContent = "85%";
    $("scopeFullPage").checked = true;
    $("detectedBadge").classList.add("hidden");
    await saveSettings(false);
    $("apiStatusText").textContent = "Reset to default profile.";
  };

  // Screening buttons
  $("retryTestBtn").onclick = startScreeningTest;

  $("applyDiagnosisBtn").onclick = async () => {
    if (latestDiagnosis && latestDiagnosis.condition_id !== null && latestDiagnosis.condition_id !== undefined) {
      $("conditionSelect").value = String(latestDiagnosis.condition_id);
      if (latestDiagnosis.severity) {
        const pct = Math.round(latestDiagnosis.severity * 100);
        $("severitySlider").value = pct;
        $("severityValue").textContent = `${pct}%`;
      }
      $("enableToggle").checked = true;
      $("detectedBadge").classList.remove("hidden");
      await saveSettings(true);
    }
    switchTab("Dashboard");
  };

  $("discardDiagnosisBtn").onclick = () => switchTab("Dashboard");

  // Initial load
  await loadSettings();
  await checkHealth();
});
