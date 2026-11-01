# Model Checkpoint Directory

Place your trained PyTorch checkpoint here:
- Recommended filename: `best_restormer.pt` (or `Modified_Conditional_Restormer_seed_123_best.pt`)

You can download this checkpoint directly from your Google Drive path:
`/content/drive/MyDrive/FYDP/ICCIT_2026_LIGHTWEIGHT_RESTORMER/results/checkpoints/Modified_Conditional_Restormer_seed_123_best.pt`

When present, the FastAPI server will automatically load these weights on startup.
If no weights file is present, the server initializes with random weights so you and your friend can still test the API and Chrome Extension communication immediately.
