"""Unified PyTorch Stage-3 training: supports MLP/Gate/Attn models, multiple losses, masking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core_modules.losses import HuberLoss, CompositeLoss, ZINBLoss
from core_modules.retrieval import MemoryBank
from metrics import compute_pcc, evaluate_predictions
from models.clip_vae import ContrastiveAutoencoder
from models.rag_flow import ResidualMLP, GateResidualMLP, ZINBResidualMLP, GateAttentionResidual, AlphaBlendResidual, FiLMResidualMLP, FiLMGateResidualMLP, FiLMRectifiedFlow, euler_integrate, apply_residual_correction
from utils import preprocess_cross_dataset, preprocess_single_dataset, save_json, set_seed, stable_cache_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--test_data_path", type=str, default="")
    p.add_argument("--stage1_path", type=str, required=True)
    p.add_argument("--bank_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_top_genes", type=int, default=1000)
    p.add_argument("--train_ratio", type=float, default=0.8)

    # Model
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "gate", "gate_attn", "zinb", "alpha_blend", "film", "film_gate", "rectified_flow"])
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--film_gate_zinb", action="store_true", help="Add ZINB heads to film_gate model")

    # Loss
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber", "composite", "zinb", "mse_zinb"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--w_cos", type=float, default=0.1)
    p.add_argument("--w_pearson", type=float, default=0.1)
    p.add_argument("--w_zinb", type=float, default=0.5)

    # Retrieval
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--k_list", type=int, nargs="+", default=[3, 5, 7])

    # Input augmentations (Exp-A, Exp-C)
    p.add_argument("--mask_prob", type=float, default=0.0, help="Mask prob for retrieved protein")
    p.add_argument("--retrieval_noise", type=float, default=0.0, help="Gaussian noise std for retrieved protein")
    p.add_argument("--bottleneck_dim", type=int, default=0, help="Dim for raw RNA bottleneck (0=disable)")
    p.add_argument("--with_raw_rna", action="store_true", help="Concatenate raw RNA features")
    p.add_argument("--with_distance", action="store_true", help="Append retrieval distance to cond")
    p.add_argument("--protein_guided_hvg", action="store_true", help="Use protein-guided HVG instead of scanpy default")
    p.add_argument("--per_protein_hvg", action="store_true", help="Use per-protein fair HVG")
    p.add_argument("--n_per_protein", type=int, default=50)
    p.add_argument("--finetune_stage1", action="store_true", help="Finetune Stage1 RNA encoder during Stage3 training")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--encode_batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fold_split_path", type=str, default="", help="Path to .npz with train_idx/test_idx for fixed split")
    return p.parse_args()


def build_model(args, cond_dim: int, protein_dim: int, raw_rna_dim: int = 0) -> nn.Module:
    if args.model == "gate":
        return GateResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
            with_raw_rna=args.with_raw_rna, raw_rna_dim=raw_rna_dim,
        )
    elif args.model == "gate_attn":
        return GateAttentionResidual(
            cond_dim=cond_dim, protein_dim=protein_dim,
            num_k=len(args.k_list), hidden_dim=min(args.hidden_dim, 128),
            num_heads=args.num_heads, depth=args.depth, dropout=args.dropout,
        )
    elif args.model == "zinb":
        return ZINBResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
        )
    elif args.model == "alpha_blend":
        return AlphaBlendResidual(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
        )
    elif args.model == "film":
        return FiLMResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
        )
    elif args.model == "film_gate":
        return FiLMGateResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
            use_zinb=args.film_gate_zinb,
        )
    elif args.model == "rectified_flow":
        return FiLMRectifiedFlow(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
        )
    else:  # mlp
        return ResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
            with_raw_rna=args.with_raw_rna, raw_rna_dim=raw_rna_dim,
        )


def build_loss(args) -> nn.Module:
    if args.loss == "huber":
        return HuberLoss(delta=args.huber_delta)
    elif args.loss == "composite":
        return CompositeLoss(w_cos=args.w_cos, w_pearson=args.w_pearson)
    elif args.loss in ("zinb", "mse_zinb"):
        return ZINBLoss(ridge_lambda=1e-4)
    else:
        return nn.MSELoss()


def compute_loss(model_out, target, loss_fn, args):
    if args.loss == "zinb":
        return loss_fn(target, model_out["mu"], model_out["theta"], model_out["pi"])
    elif args.loss == "mse_zinb":
        mse = F.mse_loss(model_out["pred"], target)
        zinb = loss_fn(target, model_out["mu"], model_out["theta"], model_out["pi"])
        return mse + args.w_zinb * zinb
    elif args.loss == "composite":
        return loss_fn(model_out, target) if isinstance(model_out, torch.Tensor) else loss_fn(model_out["pred"], target)
    else:
        return loss_fn(model_out, target) if isinstance(model_out, torch.Tensor) else loss_fn(model_out["pred"], target)


def apply_mask(start: torch.Tensor, mask_prob: float, noise_std: float = 0.0) -> torch.Tensor:
    if mask_prob > 0:
        mask = torch.rand_like(start) < mask_prob
        start = torch.where(mask, torch.zeros_like(start), start)
    if noise_std > 0:
        start = start + torch.randn_like(start) * noise_std
    return start


@torch.no_grad()
def encode_in_batches(stage1, array: np.ndarray, bs: int, device: torch.device) -> np.ndarray:
    outs = []
    for i in range(0, len(array), bs):
        e = min(i + bs, len(array))
        b = torch.from_numpy(array[i:e]).to(device, non_blocking=True)
        outs.append(stage1.encode_rna(b).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def predict(model, cond, start, raw_rna, batch_size, device, is_attn, attn_starts=None,
            bottleneck=None):
    model.eval()
    if bottleneck is not None:
        bottleneck.eval()
    preds = []
    for i in range(0, len(cond), batch_size):
        e = min(i + batch_size, len(cond))
        c = torch.from_numpy(cond[i:e]).to(device)
        s = torch.from_numpy(start[i:e]).to(device) if start is not None else None
        r = torch.from_numpy(raw_rna[i:e]).to(device) if raw_rna is not None else None
        if bottleneck is not None and r is not None:
            r = bottleneck(r)
        if is_attn and attn_starts is not None:
            a = torch.from_numpy(attn_starts[i:e]).to(device)
            out = model(a, c)
        else:
            out = apply_residual_correction(model, s, c, r)
        preds.append(out.cpu().numpy() if isinstance(out, torch.Tensor) else out)
    return np.concatenate(preds, axis=0).astype(np.float32)


def collect_attn_starts(bank, features, k_list, exclude_self):
    """Build multi-k [B, K, P] starts for gate_attn."""
    starts = []
    for k in k_list:
        result = bank.query(features, k=k, exclude_self=exclude_self)
        weights = np.exp(result.scores - result.scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8
        neighbors = bank.payload[result.indices]
        mean = np.einsum("bk,bkd->bd", weights, neighbors).astype(np.float32)
        starts.append(mean)
    return np.stack(starts, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Load Stage1
    ckpt = torch.load(args.stage1_path, map_location=device, weights_only=False)
    stage1 = ContrastiveAutoencoder(**ckpt["config"]).to(device)
    stage1.load_state_dict(ckpt["model_state_dict"])
    if not args.finetune_stage1:
        stage1.eval()
        for p in stage1.parameters():
            p.requires_grad = False
    else:
        stage1.train()

    # Load data (with optional fixed fold split)
    fold_train_idx = fold_test_idx = None
    if args.fold_split_path:
        split_data = np.load(args.fold_split_path)
        fold_train_idx = split_data["train_idx"]
        fold_test_idx = split_data["test_idx"]
    # Use Stage1's saved gene_names to skip HVG recomputation (fast!)
    s1_gene_names = ckpt.get('gene_names', None)
    if args.test_data_path:
        bundle = preprocess_cross_dataset(args.data_path, args.test_data_path, n_top_genes=args.n_top_genes)
    else:
        bundle = preprocess_single_dataset(
            args.data_path, n_top_genes=args.n_top_genes, train_ratio=args.train_ratio, seed=args.seed,
            use_protein_guided_hvg=args.protein_guided_hvg,
            use_per_protein_hvg=args.per_protein_hvg,
            n_per_protein=args.n_per_protein,
            fold_train_idx=fold_train_idx, fold_test_idx=fold_test_idx,
            gene_names=s1_gene_names,
        )

    # Load bank
    bank_data = np.load(args.bank_path, allow_pickle=True)
    bank = MemoryBank()
    bank.fit(bank_data["features"], bank_data["payload"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cond_dim = ckpt["config"]["latent_dim"]
    cond_dim = base_cond_dim + (1 if args.with_distance else 0)
    protein_dim = len(bundle.protein_names)
    raw_rna_dim = bundle.rna_norm_train.shape[1] if args.with_raw_rna else 0
    is_attn = args.model == "gate_attn"
    is_flow = args.model == "rectified_flow"

    # Cache features
    cache_root = output_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = stable_cache_key(
        args.data_path, args.test_data_path, args.stage1_path, args.bank_path,
        args.n_top_genes, args.train_ratio, args.k if not is_attn else tuple(args.k_list),
        args.with_raw_rna, args.with_distance, args.protein_guided_hvg,
        args.per_protein_hvg, args.n_per_protein,
        len(bundle.train_idx), len(bundle.test_idx),
    )
    cache_path = cache_root / f"torch_features_{cache_key}.npz"

    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        train_cond = cache["train_cond"]
        test_cond = cache["test_cond"]
        train_start = cache["train_start"]
        test_start = cache["test_start"]
        train_raw = cache.get("train_raw", None)
        test_raw = cache.get("test_raw", None)
        train_dist = cache.get("train_dist", None)
        test_dist = cache.get("test_dist", None)
        train_attn_starts = cache.get("train_attn_starts", None)
        test_attn_starts = cache.get("test_attn_starts", None)
        print(f"Loaded cache: {cache_path}", flush=True)
    else:
        train_cond = encode_in_batches(stage1, bundle.rna_norm_train, args.encode_batch_size, device)
        test_cond = encode_in_batches(stage1, bundle.rna_norm_test, args.encode_batch_size, device)

        if is_attn:
            train_start = bank.aggregate(train_cond, k=args.k_list[0], exclude_self=True)
            test_start = bank.aggregate(test_cond, k=args.k_list[0], exclude_self=False)
            train_attn_starts = collect_attn_starts(bank, train_cond, args.k_list, exclude_self=True)
            test_attn_starts = collect_attn_starts(bank, test_cond, args.k_list, exclude_self=False)
        else:
            qk = args.k
            train_result = bank.query(train_cond, k=qk, exclude_self=True)
            test_result = bank.query(test_cond, k=qk, exclude_self=False)
            train_start = bank.aggregate(train_cond, k=qk, exclude_self=True)
            test_start = bank.aggregate(test_cond, k=qk, exclude_self=False)
            train_dist = train_result.scores[:, 0:1].astype(np.float32)  # closest neighbor similarity
            test_dist = test_result.scores[:, 0:1].astype(np.float32)
            train_attn_starts = test_attn_starts = None

        save_dict = dict(train_cond=train_cond, test_cond=test_cond,
                         train_start=train_start, test_start=test_start)
        if args.with_raw_rna:
            train_raw = bundle.rna_norm_train.astype(np.float32)
            test_raw = bundle.rna_norm_test.astype(np.float32)
            save_dict["train_raw"] = train_raw
            save_dict["test_raw"] = test_raw
        if args.with_distance:
            save_dict["train_dist"] = train_dist
            save_dict["test_dist"] = test_dist
        if is_attn:
            save_dict["train_attn_starts"] = train_attn_starts
            save_dict["test_attn_starts"] = test_attn_starts
        np.savez_compressed(cache_path, **save_dict)
        print(f"Saved cache: {cache_path}", flush=True)

    # Build model (with optional bottleneck for Exp-C)
    effective_raw_dim = args.bottleneck_dim if (args.with_raw_rna and args.bottleneck_dim > 0) else raw_rna_dim
    model = build_model(args, cond_dim, protein_dim, effective_raw_dim).to(device)
    bottleneck = None
    if args.with_raw_rna and args.bottleneck_dim > 0 and raw_rna_dim > 0:
        bottleneck = nn.Sequential(
            nn.Linear(raw_rna_dim, args.bottleneck_dim),
            nn.LayerNorm(args.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        ).to(device)
    params = list(model.parameters())
    if bottleneck is not None:
        params += list(bottleneck.parameters())
    if args.finetune_stage1:
        params += list(stage1.parameters())
        print(f"  Finetuning Stage1: adding {sum(p.numel() for p in stage1.parameters())} params", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_loss(args)

    # Concat distance to cond if enabled
    if args.with_distance and train_dist is not None:
        train_cond = np.concatenate([train_cond, train_dist], axis=1)
        test_cond = np.concatenate([test_cond, test_dist], axis=1)

    # Dataset
    if args.finetune_stage1 and not is_attn:
        # Use raw RNA norm for on-the-fly encoding; start is pre-computed from initial Stage1
        train_tensors = [torch.from_numpy(bundle.rna_norm_train.astype(np.float32)),
                         torch.from_numpy(train_start),
                         torch.from_numpy(bundle.protein_norm_train.astype(np.float32))]
        if args.with_raw_rna and train_raw is not None:
            train_tensors.append(torch.from_numpy(train_raw))
    elif is_attn:
        train_tensors = [torch.from_numpy(train_cond), torch.from_numpy(train_attn_starts),
                         torch.from_numpy(bundle.protein_norm_train.astype(np.float32))]
    else:
        train_tensors = [torch.from_numpy(train_cond), torch.from_numpy(train_start),
                         torch.from_numpy(bundle.protein_norm_train.astype(np.float32))]
        if args.with_raw_rna and train_raw is not None:
            train_tensors.append(torch.from_numpy(train_raw))

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")

    print(f"Model: {args.model}, Loss: {args.loss}, Mask: {args.mask_prob}, "
          f"RawRNA: {args.with_raw_rna}, Params: {sum(p.numel() for p in model.parameters())}", flush=True)

    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bottleneck is not None:
            bottleneck.train()
        total_loss = 0.0
        n_batch = 0

        for batch in train_loader:
            if is_attn:
                cond_b, starts_b, target_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                start_b = None; raw_b = None
            elif args.finetune_stage1:
                # On-the-fly RNA encoding: [rna_norm, start, target, optional_raw]
                rna_b = batch[0].to(device)
                start_b = batch[1].to(device)
                target_b = batch[2].to(device)
                cond_b = stage1.encode_rna(rna_b)
                starts_b = None
                raw_b = batch[3].to(device) if len(batch) > 3 and args.with_raw_rna else None
            else:
                cond_b, start_b, target_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                starts_b = None
                raw_b = batch[3].to(device) if len(batch) > 3 and args.with_raw_rna else None

            # Apply retrieval dropout/noise + mask (Exp-A)
            # CRITICAL: save original start for computing v_true (velocity target).
            # Masked start is fed to model as input; original start is used for target residual.
            start_original = start_b.clone() if start_b is not None else None
            if (args.mask_prob > 0 or args.retrieval_noise > 0) and start_b is not None:
                start_b = apply_mask(start_b, args.mask_prob, args.retrieval_noise)

            # Apply bottleneck to raw RNA (Exp-C)
            if bottleneck is not None and raw_b is not None:
                raw_b = bottleneck(raw_b)

            # Forward
            if is_flow:
                # Rectified Flow: predict velocity v = x1 - x0
                x_0 = start_b + torch.randn_like(start_b) * 0.2  # noise injection
                x_1 = target_b
                t_vals = torch.rand(x_0.size(0), 1, device=device)
                x_t = (1 - t_vals) * x_0 + t_vals * x_1
                v_pred = model(x_t, t_vals.squeeze(-1), cond_b)
                # v_true computed from original start (unmasked + same noise)
                x_0_orig = (start_original if start_original is not None else start_b) + torch.randn_like(start_b) * 0.2
                v_true = x_1 - x_0_orig
                loss = F.mse_loss(v_pred, v_true)
                n_batch += 1; total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                continue
            elif is_attn:
                out = model(starts_b, cond_b)
                loss_target = target_b
            elif isinstance(model, (AlphaBlendResidual,)):
                out = model(start_b, cond_b, raw_b)
                loss_target = target_b
            elif isinstance(model, FiLMGateResidualMLP):
                out = model(start_b, cond_b, raw_b)
                loss_target = target_b  # FiLMGate returns absolute pred (or dict)
            elif isinstance(model, (ResidualMLP, FiLMResidualMLP)):
                # ── Configurable Flow Matching ──
                # TRAJECTORY_MODE: "none" (no t) | "fixed" (t=0) | "full" (t~U[0,1])
                TRAJECTORY_MODE = "none"
                x_0, x_1 = start_b, target_b
                v_true = x_1 - (start_original if start_original is not None else x_0)
                if TRAJECTORY_MODE == "full":
                    tv = torch.rand(x_0.shape[0], device=device)
                    x_t = (1 - tv.unsqueeze(1)) * x_0 + tv.unsqueeze(1) * x_1
                    v_pred = model(x_t, cond_b, raw_b, t=tv)
                elif TRAJECTORY_MODE == "fixed":
                    tv = torch.zeros(x_0.shape[0], device=device)
                    v_pred = model(x_0, cond_b, raw_b, t=tv)
                else:  # "none"
                    v_pred = model(x_0, cond_b, raw_b)
                loss = F.mse_loss(v_pred, v_true)
                n_batch += 1; total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                continue
            else:
                # Gate, ZINB etc. handle their own output space
                out = model(start_b, cond_b, raw_b)
                loss_target = target_b

            loss = compute_loss(out, loss_target, loss_fn, args)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        train_loss = total_loss / max(n_batch, 1)

        # Per-epoch log (so we can see progress)
        if not args.finetune_stage1:
            # fast path: no on-the-fly encoding, print every epoch
            pass  # printed below
        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if not should_eval:
            print(f"Epoch {epoch:03d} | train={train_loss:.4f}", flush=True)

        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            if args.finetune_stage1:
                # Recompute test cond with updated Stage1
                test_cond = encode_in_batches(stage1, bundle.rna_norm_test, args.encode_batch_size, device)
                if args.with_distance and test_dist is not None:
                    test_cond_eval = np.concatenate([test_cond, test_dist], axis=1)
                else:
                    test_cond_eval = test_cond
            else:
                test_cond_eval = test_cond
            preds = predict(model, test_cond_eval, test_start, test_raw if args.with_raw_rna else None,
                           args.batch_size, device, is_attn,
                           test_attn_starts if is_attn else None,
                           bottleneck=bottleneck)
            if preds is not None:
                pcc_p, pcc_c, pcc_pm, pcc_cm = compute_pcc(preds, bundle.protein_norm_test, bundle.protein_names)
                results = evaluate_predictions(preds, bundle.protein_norm_test, bundle.protein_names, verbose=False)
                print(f"Epoch {epoch:03d} | train={train_loss:.4f} | protein_pcc={pcc_pm:.4f} cell_pcc={pcc_cm:.4f}", flush=True)

                save_json(output_dir / "metrics_last.json", {
                    "epoch": epoch, "train_loss": train_loss,
                    "pcc_protein_mean": float(pcc_pm), "pcc_protein_median": float(np.nanmedian(pcc_p)) if isinstance(pcc_p, np.ndarray) else 0.0,
                    "pcc_cell_mean": float(pcc_cm),
                })

                # Save last
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "config": vars(args)}, output_dir / "model_last.pt")

                if pcc_pm > best_score:
                    best_score = pcc_pm
                    best_epoch = epoch
                    np.save(output_dir / "predictions.npy", preds)
                    np.save(output_dir / "ground_truth.npy", bundle.protein_norm_test)
                    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                                "config": vars(args)}, output_dir / "model_best.pt")
                    save_json(output_dir / "metrics_best.json",
                              {k: float(v) if np.isscalar(v) else None
                               for k, v in results.items() if "pcc_" not in k or k.endswith(("mean", "median"))})
        else:
            if best_score < 0:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "config": vars(args)}, output_dir / "model_best.pt")

    # Final eval
    best_ckpt = torch.load(output_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    if args.finetune_stage1:
        test_cond = encode_in_batches(stage1, bundle.rna_norm_test, args.encode_batch_size, device)
        if args.with_distance and test_dist is not None:
            test_cond_eval = np.concatenate([test_cond, test_dist], axis=1)
        else:
            test_cond_eval = test_cond
    else:
        test_cond_eval = test_cond
    preds = predict(model, test_cond_eval, test_start, test_raw if args.with_raw_rna else None,
                   args.batch_size, device, is_attn,
                   test_attn_starts if is_attn else None,
                   bottleneck=bottleneck)
    if preds is not None:
        final = evaluate_predictions(preds, bundle.protein_norm_test, bundle.protein_names, verbose=True)
        print(f"Best at epoch {best_epoch}: protein_pcc={final['pcc_protein_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
