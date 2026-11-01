# Personalized CVD Real-Time Web Recoloring Assistant

A research-grounded, deep-learning powered accessibility system designed to assist users with Color Vision Deficiency (CVD) — Protanopia (red-weak), Deuteranopia (green-weak), and Tritanopia (blue-weak) — while browsing the modern web.

---

## System Architecture

```text
Chrome Extension (Manifest V3)
        ↓
CVD Screening / Profile Estimation (or manual selection)
        ↓
Protanopia (0) / Deuteranopia (1) / Tritanopia (2) + Severity [0.0 - 1.0]
        ↓
Locally saved via chrome.storage.local
        ↓
User browses webpage → content.js detects & filters eligible images
        ↓
Background Service Worker fetches & proxies request
        ↓
FastAPI Backend (POST /api/v1/recolor)
        ↓
Restormer_RDB (Ablation Peak Production Model)
        ↓
Personalization Blending: Output = severity * Model + (1 - severity) * Original
        ↓
Corrected WebP / PNG / JPEG visual returned
        ↓
Non-destructive DOM image replacement preserving layout
```

---

## 1. Selected Production Model

- **Architecture**: `Restormer_RDB`
- **Ablation Benchmark Evidence**: Evaluated against 6 architectures in research notebook `a-lightweight-conditional-restormer-for-multi-type (1).ipynb`:
  - `Restormer_RDB`: **22.3635 dB PSNR**, **0.8929 MS-SSIM**, **0.1088 LPIPS** (461,694 params, 7.4ms latency) — *Highest reconstruction fidelity & structural similarity*
  - `Modified_Conditional_Restormer`: 22.1116 dB PSNR, 0.8898 MS-SSIM, 0.1135 LPIPS (462,888 params)
  - `Conditional_Residual_UNet`: 21.9405 dB PSNR, 0.8835 MS-SSIM, 0.1213 LPIPS (1,927,939 params)
  - `Plain_Conditional_Residual_CNN`: 22.0120 dB PSNR, 0.8812 MS-SSIM, 0.1318 LPIPS (337,251 params)
- **Input Channels**: 6 (RGB 3-channel + spatially broadcasted one-hot condition map (B, 3, H, W))
- **Condition Mechanism**: Spatially broadcasted condition tensor concatenated with input image
- **Severity Mechanism**: Post-inference continuous blending:
  $$\text{Final} = \text{severity} \times \text{ModelOutput} + (1 - \text{severity}) \times \text{Original}$$

---

## 2. Dataset Protocol & Leakage Prevention

- **Dataset Root**: `archive/data/`
  - Reference Originals: 858 images (`original/`)
  - Corrected Targets: 858 images per condition (`corrected/protanopia`, `corrected/deuteranopia`, `corrected/tritanopia`)
  - Total Verified Pairs: 2,574 condition-specific pairs (858 × 3)
- **Identity-Grouped Split (Seed 2026)**:
  - **Train (70%)**: 601 identities | 1,803 image pairs
  - **Validation (15%)**: 129 identities | 387 image pairs
  - **Test (15%)**: 128 identities | 384 image pairs
- **Zero Leakage**: All condition pairs belonging to an original identity are confined strictly to one split. Verified with automated assertions.

---

## 3. Environment Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Model Training & Resume

### Training from Scratch:
```bash
backend/venv/bin/python backend/train_or_eval.py \
  --epochs 60 \
  --batch-size 16 \
  --lr 3e-4 \
  --model-name Restormer_RDB \
  --data-root archive/data \
  --weights-out backend/weights/best_restormer.pt
```

### Resume Training:
```bash
backend/venv/bin/python backend/train_or_eval.py \
  --resume backend/weights/last_checkpoint.pt \
  --epochs 60 \
  --batch-size 16 \
  --model-name Restormer_RDB
```

---

## 5. Model Evaluation

Evaluates model on the 128 held-out test identities (384 pairs) with overall and per-condition breakdown (PSNR, SSIM, MS-SSIM, MAE, MSE, latency):

```bash
backend/venv/bin/python backend/train_or_eval.py \
  --eval-only \
  --checkpoint backend/weights/best_restormer.pt \
  --model-name Restormer_RDB \
  --batch-size 16
```

---

## 6. Running FastAPI Server

```bash
cd backend
./venv/bin/python run_server.py
```
- API Base: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Recolor Endpoint: `POST http://localhost:8000/api/v1/recolor`

---

## 7. Automated Test Suite

Runs the comprehensive 24-test verification suite covering API endpoints, error handling, format support, preprocessing padding, cache isolation, and zero-leakage assertions:

```bash
backend/venv/bin/python -m unittest backend/tests/test_master_suite.py -v
```

---

## 8. Chrome Extension Installation (Manifest V3)

1. Open Google Chrome.
2. Navigate to `chrome://extensions/`.
3. Enable **Developer mode** via the top-right toggle switch.
4. Click **Load unpacked** in the top-left toolbar.
5. Select the `extension/` directory from this repository.
6. Click the extension icon in the toolbar to open the popup:
   - Run the **Profile Estimation** test or select your profile manually.
   - Adjust the **Personalization Severity** slider (0% to 100%).
   - Switch the toggle **ON** to enable real-time per-image web recoloring.
   - Open `extension/test_page.html` in Chrome to observe live image transformation and icon filtering.
