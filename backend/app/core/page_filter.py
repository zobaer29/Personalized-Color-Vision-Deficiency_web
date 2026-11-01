"""Fit a global sRGB affine approximation to trained model swatch responses.
This is a browser acceleration approximation, never pixelwise Restormer inference.
"""
from functools import lru_cache
import numpy as np
import torch

@lru_cache(maxsize=12)
def fit_filter(predictor, fingerprint, condition_id):
    if not predictor.weights_loaded:
        raise RuntimeError('Trained Restormer_RDB checkpoint is required')
    colors = np.array([[r,g,b] for r in np.linspace(0,1,5)
                      for g in np.linspace(0,1,5) for b in np.linspace(0,1,5)], dtype=np.float32)
    outputs = []
    with predictor.lock, torch.inference_mode():
        for batch in np.array_split(colors, 9):
            x = torch.tensor(batch, device=predictor.device)[:,:,None,None].expand(-1,-1,16,16)
            c = torch.full((len(batch),), condition_id, device=predictor.device, dtype=torch.long)
            outputs.append(predictor.model(x,c)[:,:,4:-4,4:-4].mean((2,3)).cpu().numpy())
    targets = np.concatenate(outputs)
    design = np.column_stack([colors, np.ones(len(colors))])
    coeff = np.linalg.lstsq(design, targets, rcond=None)[0]
    error = float(np.sqrt(np.mean((design @ coeff - targets)**2)))
    return coeff, error

def page_filter(predictor, condition_id, strength):
    coeff, error = fit_filter(predictor, predictor.fingerprint, condition_id)
    matrix = np.zeros((4,5), dtype=float)
    matrix[:3,:3] = coeff[:3].T * strength + np.eye(3)*(1-strength)
    matrix[:3,4] = coeff[3]*strength
    matrix[3,3] = 1
    return dict(matrix=matrix.flatten().tolist(), model=predictor.model_name,
                checkpoint_sha256=predictor.fingerprint, approximation=True,
                calibration_rmse=error, strength=strength)
