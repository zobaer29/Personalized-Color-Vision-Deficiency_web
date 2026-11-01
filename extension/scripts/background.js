/**
 * Manifest V3 Service Worker for Personalized CVD Browser Assistant.
 * Handles:
 * - Cross-origin image fetching & Base64 conversion
 * - Proxying inference requests to FastAPI /api/v1/recolor
 * - In-flight request deduplication & error handling
 * - Storage initialization
 */

const API_BASE = "http://localhost:8000";

// Initialize default storage on install
chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope"]);
  const updates = {};
  if (current.enabled === undefined) updates.enabled = false;
  if (current.condition_id === undefined) updates.condition_id = 1; // Default Deuteranopia
  if (current.severity === undefined) updates.severity = 0.85;
  if (current.scope === undefined) updates.scope = "full_page";
  if (Object.keys(updates).length > 0) {
    await chrome.storage.local.set(updates);
  }
});

// Cache for in-flight requests to deduplicate concurrent requests for identical images
const inFlightRequests = new Map();

/**
 * Converts a Blob to a Base64 data URL string.
 */
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Fetches an image URL and converts it to a Base64 data URL.
 */
async function fetchImageAsBase64(imageUrl) {
  if (imageUrl.startsWith("data:image/")) {
    return imageUrl;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch(imageUrl, {
      signal: controller.signal,
      cache: "force-cache",
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Failed to fetch image: HTTP ${response.status}`);
    }

    const blob = await response.blob();
    return await blobToBase64(blob);
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Sends image data URL to the FastAPI recolor endpoint.
 */
async function sendRecolorRequest(imageBase64, conditionId, severity) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch(`${API_BASE}/api/v1/recolor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        image_base64: imageBase64,
        condition_id: conditionId,
        severity: severity,
        return_format: "webp",
      }),
    });
    clearTimeout(timeoutId);

    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `API Error ${response.status}`);
    }

    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// Handle messages from content script & popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "RECOLOR_IMAGE") {
    const { src, conditionId, severity } = message;
    const cacheKey = `${src}|${conditionId}|${severity}`;

    if (!inFlightRequests.has(cacheKey)) {
      const task = (async () => {
        try {
          const base64Data = await fetchImageAsBase64(src);
          const recolorResult = await sendRecolorRequest(base64Data, conditionId, severity);
          return {
            success: true,
            recolored_base64: recolorResult.recolored_base64,
            inference_time_ms: recolorResult.inference_time_ms,
            cached: recolorResult.cached,
          };
        } catch (error) {
          return {
            success: false,
            error: error.message || "Recoloring failed",
          };
        }
      })();

      inFlightRequests.set(cacheKey, task);
      task.finally(() => inFlightRequests.delete(cacheKey));
    }

    inFlightRequests.get(cacheKey).then(sendResponse);
    return true; // Keep channel open for async response
  }

  if (message.action === "CHECK_HEALTH") {
    fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.action === "GET_FILTER") {
    fetch(`${API_BASE}/api/v1/filter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(10000),
      body: JSON.stringify({ condition_id: message.conditionId, severity: message.severity }),
    })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  return false;
});
