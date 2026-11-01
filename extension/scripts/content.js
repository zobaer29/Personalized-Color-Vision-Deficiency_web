/**
 * Content Script for Personalized CVD Web Assistant.
 * Scans page images, filters eligible graphics, queues inference requests,
 * and performs non-destructive image replacements while preserving layout.
 */

(() => {
  "use strict";

  // Prevent double-injection
  if (window.__cvdAssistantInjected) return;
  window.__cvdAssistantInjected = true;

  const STATUS_ATTR = "data-cvd-status";
  const ORIG_SRC_ATTR = "data-cvd-orig-src";
  const PROFILE_KEY_ATTR = "data-cvd-profile";

  const FILTER_SVG_ID = "cvd-assistant-svg-filter";
  const FILTER_ID = "cvd-assistant-matrix";
  const FILTER_STYLE_ID = "cvd-assistant-filter-style";

  let currentSettings = {
    enabled: false,
    condition_id: 1,
    severity: 0.85,
    scope: "full_page", // "full_page" (entire website) or "images_only" (photos only)
  };

  let cachedFilterMatrix = null;
  let cachedFilterKey = "";

  /**
   * Injects/updates an SVG color matrix filter into the document and applies it to <html>.
   * This recolors the entire website (CSS backgrounds, text, UI elements, palettes, images).
   */
  function applyPageFilter(matrixValues) {
    if (!matrixValues || matrixValues.length !== 20) return;

    let svg = document.getElementById(FILTER_SVG_ID);
    if (!svg) {
      svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.id = FILTER_SVG_ID;
      svg.setAttribute("style", "position:fixed; top:0; left:0; width:0; height:0; pointer-events:none; z-index:-1000; opacity:0;");
      svg.setAttribute("aria-hidden", "true");
      svg.innerHTML = `
        <defs>
          <filter id="${FILTER_ID}" color-interpolation-filters="sRGB">
            <feColorMatrix type="matrix" values="${matrixValues.join(' ')}" />
          </filter>
        </defs>
      `;
      (document.body || document.documentElement).appendChild(svg);
    } else {
      const feMatrix = svg.querySelector("feColorMatrix");
      if (feMatrix) {
        feMatrix.setAttribute("values", matrixValues.join(" "));
      }
    }

    let style = document.getElementById(FILTER_STYLE_ID);
    if (!style) {
      style = document.createElement("style");
      style.id = FILTER_STYLE_ID;
      document.head.appendChild(style);
    }
    style.textContent = `
      html {
        filter: url(#${FILTER_ID}) !important;
      }
    `;
  }

  /**
   * Removes the full-page SVG filter and restores standard rendering.
   */
  function removePageFilter() {
    const style = document.getElementById(FILTER_STYLE_ID);
    if (style) style.remove();
    const svg = document.getElementById(FILTER_SVG_ID);
    if (svg) svg.remove();
  }

  /**
   * Synchronizes full-page SVG filter with the backend.
   */
  async function syncPageFilter() {
    if (!currentSettings.enabled || currentSettings.scope === "images_only") {
      removePageFilter();
      return;
    }

    const filterKey = `${currentSettings.condition_id}|${currentSettings.severity}`;
    if (cachedFilterMatrix && cachedFilterKey === filterKey) {
      applyPageFilter(cachedFilterMatrix);
      return;
    }

    try {
      const response = await chrome.runtime.sendMessage({
        action: "GET_FILTER",
        conditionId: currentSettings.condition_id,
        severity: currentSettings.severity,
      });

      if (response && response.success && response.data?.matrix) {
        cachedFilterMatrix = response.data.matrix;
        cachedFilterKey = filterKey;
        applyPageFilter(cachedFilterMatrix);
      }
    } catch (err) {
      console.warn("[CVD Assistant] Failed to retrieve whole-page filter matrix:", err);
    }
  }

  // Queue and concurrency controller (max 2 concurrent requests to avoid server overloading)
  const MAX_CONCURRENT = 2;
  let activeRequests = 0;
  const requestQueue = [];

  function enqueue(task) {
    requestQueue.push(task);
    processNext();
  }

  function processNext() {
    if (activeRequests >= MAX_CONCURRENT || requestQueue.length === 0) return;
    activeRequests++;
    const task = requestQueue.shift();
    task().finally(() => {
      activeRequests--;
      processNext();
    });
  }

  /**
   * Evaluates whether an <img> is eligible for neural recoloring.
   */
  function isEligibleImage(img) {
    if (!img || !(img instanceof HTMLImageElement)) return false;

    // Check dimensions (skip tiny icons, tracking pixels)
    const rect = img.getBoundingClientRect();
    const w = rect.width || img.naturalWidth || img.width;
    const h = rect.height || img.naturalHeight || img.height;

    if (w < 32 || h < 32) return false;

    const src = img.currentSrc || img.src;
    if (!src) return false;

    // Skip extension assets and SVG data URLs
    if (src.startsWith("chrome-extension://") || src.startsWith("moz-extension://")) return false;
    if (src.startsWith("data:image/svg+xml")) return false;

    // Skip already processed images with same profile
    const profileKey = `${currentSettings.condition_id}|${currentSettings.severity}`;
    if (img.getAttribute(STATUS_ATTR) === "processed" && img.getAttribute(PROFILE_KEY_ATTR) === profileKey) {
      return false;
    }

    if (img.getAttribute(STATUS_ATTR) === "pending") return false;

    return true;
  }

  /**
   * Processes a single eligible image.
   */
  async function processImage(img) {
    if (!currentSettings.enabled || !isEligibleImage(img)) return;

    const src = img.currentSrc || img.src;
    if (!src) return;

    const profileKey = `${currentSettings.condition_id}|${currentSettings.severity}`;

    // Save original source for non-destructive restore
    if (!img.hasAttribute(ORIG_SRC_ATTR)) {
      img.setAttribute(ORIG_SRC_ATTR, src);
    }

    // Preserve original layout dimensions
    const computedStyle = window.getComputedStyle(img);
    if (!img.style.width && computedStyle.width) {
      img.style.width = computedStyle.width;
    }
    if (!img.style.height && computedStyle.height) {
      img.style.height = computedStyle.height;
    }

    img.setAttribute(STATUS_ATTR, "pending");

    enqueue(async () => {
      if (!currentSettings.enabled) {
        img.removeAttribute(STATUS_ATTR);
        return;
      }

      try {
        const response = await chrome.runtime.sendMessage({
          action: "RECOLOR_IMAGE",
          src: img.getAttribute(ORIG_SRC_ATTR) || src,
          conditionId: currentSettings.condition_id,
          severity: currentSettings.severity,
        });

        if (!currentSettings.enabled) return;

        if (response && response.success && response.recolored_base64) {
          // Replace source non-destructively
          img.src = response.recolored_base64;
          img.setAttribute(STATUS_ATTR, "processed");
          img.setAttribute(PROFILE_KEY_ATTR, profileKey);
        } else {
          // Failure: keep original image untouched
          img.setAttribute(STATUS_ATTR, "failed");
          console.warn("[CVD Assistant] Recoloring failed for image, kept original:", response?.error);
        }
      } catch (err) {
        img.setAttribute(STATUS_ATTR, "failed");
        console.warn("[CVD Assistant] Communication error, kept original:", err.message);
      }
    });
  }

  /**
   * Scans all images within a root element.
   */
  function scanImages(root = document.body) {
    if (!root || !currentSettings.enabled) return;
    const images = root.querySelectorAll ? root.querySelectorAll("img") : [];
    images.forEach(img => {
      if (img.complete && img.naturalWidth > 0) {
        processImage(img);
      } else {
        img.addEventListener("load", () => processImage(img), { once: true });
      }
    });
  }

  /**
   * Restores all images back to their original sources when disabled.
   */
  function restoreOriginalImages() {
    requestQueue.length = 0; // Clear pending queue
    const processedImages = document.querySelectorAll(`img[${STATUS_ATTR}]`);
    processedImages.forEach(img => {
      const orig = img.getAttribute(ORIG_SRC_ATTR);
      if (orig && img.src !== orig) {
        img.src = orig;
      }
      img.removeAttribute(STATUS_ATTR);
      img.removeAttribute(PROFILE_KEY_ATTR);
    });
  }

  /**
   * Dynamic content monitoring via MutationObserver.
   */
  const observer = new MutationObserver(mutations => {
    if (!currentSettings.enabled) return;

    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.tagName === "IMG") {
            processImage(node);
          } else if (node.querySelectorAll) {
            scanImages(node);
          }
        }
      }
    }
  });

  async function updateSettings() {
    const prefs = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope"]);
    const wasEnabled = currentSettings.enabled;
    const prevScope = currentSettings.scope;
    const prevKey = `${currentSettings.condition_id}|${currentSettings.severity}|${currentSettings.scope}`;

    currentSettings.enabled = prefs.enabled ?? false;
    currentSettings.condition_id = prefs.condition_id ?? 1;
    currentSettings.severity = prefs.severity ?? 0.85;
    currentSettings.scope = prefs.scope ?? "full_page"; // Default to full_page for full website recoloring

    const newKey = `${currentSettings.condition_id}|${currentSettings.severity}|${currentSettings.scope}`;

    if (!currentSettings.enabled) {
      observer.disconnect();
      removePageFilter();
      restoreOriginalImages();
    } else {
      if (currentSettings.scope === "full_page") {
        // Full Website Mode: Apply SVG matrix to <html> so background, text, UI & images recolor together
        observer.disconnect();
        restoreOriginalImages(); // Restore original img sources so they aren't double-filtered
        await syncPageFilter();
      } else {
        // Photos Only Mode: Remove page filter, run Restormer per-image inference
        removePageFilter();
        observer.observe(document.documentElement, { childList: true, subtree: true });
        if (!wasEnabled || prevScope !== "images_only" || prevKey !== newKey) {
          if (prevKey !== newKey) {
            document.querySelectorAll(`img[${STATUS_ATTR}="processed"]`).forEach(img => {
              img.removeAttribute(STATUS_ATTR);
            });
          }
          scanImages();
        }
      }
    }
  }

  // Listen for storage changes
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && ("enabled" in changes || "condition_id" in changes || "severity" in changes || "scope" in changes)) {
      updateSettings();
    }
  });

  // Listen for explicit ping messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "SETTINGS_UPDATED") {
      updateSettings().then(() => sendResponse({ ok: true }));
      return true;
    }
  });

  // Initialize
  updateSettings();
})();
