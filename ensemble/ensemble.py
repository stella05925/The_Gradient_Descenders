import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast


def load_embedding_csv(path):
    df = pd.read_csv(path)
    emb_list = []
    for s in df["embedding"]:
        arr = np.array(ast.literal_eval(s), dtype=np.float32)
        emb_list.append(arr)
    embds = np.stack(emb_list, axis=0)
    ids = df["id"].values
    return ids, embds


def safe_ensemble(pred_flow, pred_procrustes, w1=0.6, w2=0.4, normalize=True):
    if isinstance(pred_flow, np.ndarray):
        pred_flow = torch.from_numpy(pred_flow).float()
    if isinstance(pred_procrustes, np.ndarray):
        pred_procrustes = torch.from_numpy(pred_procrustes).float()

    pred_flow_norm = F.normalize(pred_flow, p=2, dim=-1)
    pred_procrustes_norm = F.normalize(pred_procrustes, p=2, dim=-1)
    
    assert abs(w1 + w2 - 1.0) < 1e-6, "Weights must sum to 1"
    ensemble = w1 * pred_flow_norm + w2 * pred_procrustes_norm

    if normalize:
        ensemble = F.normalize(ensemble, p=2, dim=-1)
        print(f"Ensemble norms - min: {torch.norm(ensemble, p=2, dim=-1).min():.4f}, "
              f"max: {torch.norm(ensemble, p=2, dim=-1).max():.4f}, "
              f"mean: {torch.norm(ensemble, p=2, dim=-1).mean():.4f}")
    
    return ensemble


if __name__ == "__main__":
    import sys
    sys.path.append('challenge')
    from src.common import load_data, generate_submission, DEVICE

    print("Loading test data...")
    test_data = load_data("data/test/test.clean.npz")
    sample_ids = test_data["captions/ids"]

    print("\nLoading flow_matching CSV...")
    ids_flow, pred_flow = load_embedding_csv("submission-flow(1).csv")

    print("Loading procrustes CSV...")
    ids_proc, pred_proc = load_embedding_csv("submission_l2.csv")

    print("\nEnsembling with fixed weights...")
    ensemble = safe_ensemble(pred_flow, pred_proc, w1=0.4, w2=0.6)

    print("\nGenerating submission...")
    ensemble_np = ensemble.cpu().numpy() if torch.is_tensor(ensemble) else ensemble

    submission = generate_submission(sample_ids, ensemble_np, "submission_ensemble.csv")
    print("✓ Saved submission to submission_ensemble.csv")
