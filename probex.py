from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Probe modes (for X in R^{d_H x d_W})
# =============================================================================
# X: (B, d_H, d_W)
#
# Supported mode: concat_all4_bigenc_proj4
#   - Uses 4 branches: xu, xxtu, xtu, xtxu
#   - Each branch has its own projection
#   - Concatenated and passed through a big encoder
#
# Branch definitions:
#   - xu   : S = X U,       U in R^{d_W x r}  => S: (B, d_H, r)
#   - xxtu : S = (X X^T) U, U in R^{d_H x r}  => S: (B, d_H, r)
#   - xtu  : S = X^T U,     U in R^{d_H x r}  => S: (B, d_W, r)
#   - xtxu : S = (X^T X) U, U in R^{d_W x r}  => S: (B, d_W, r)


def _standardize_per_sample(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample (per batch element) standardization over (N_points, r)."""
    mean = S.mean(dim=(1, 2), keepdim=True)
    std = S.std(dim=(1, 2), keepdim=True)
    return (S - mean) / (std + eps)


class ProbeXCore(nn.Module):
    """Core ProbeX encoder with 4 branches: xu, xxtu, xtu, xtxu."""

    def __init__(
        self,
        input_shape,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__()
        self.x_center = x_center
        self.x_row_norm = x_row_norm

        d_H, d_W = int(input_shape[0]), int(input_shape[1])

        # 4 probe matrices: xu, xxtu, xtu, xtxu
        self.probes_xu = nn.Linear(n_probes, d_W, bias=False)
        self.probes_xxtu = nn.Linear(n_probes, d_H, bias=False)
        self.probes_xtu = nn.Linear(n_probes, d_H, bias=False)
        self.probes_xtxu = nn.Linear(n_probes, d_W, bias=False)

        # 4 projection layers (one per branch)
        self.shared_probe_proj_row = nn.Linear(d_H, proj_dim, bias=False)   # for xu
        self.shared_probe_proj_row2 = nn.Linear(d_H, proj_dim, bias=False)  # for xxtu
        self.shared_probe_proj_col = nn.Linear(d_W, proj_dim, bias=False)   # for xtu
        self.shared_probe_proj_col2 = nn.Linear(d_W, proj_dim, bias=False)  # for xtxu

        # Big encoder after concatenating 4 projected flats
        self.concat_all4_encoder = nn.Linear(4 * proj_dim * n_probes, rep_dim)

        # output rep dim
        self.rep_dim_out = int(rep_dim)

    # ---------------------------------------------------------------------
    # preprocessing (centering / row-normalization on the point cloud)
    # ---------------------------------------------------------------------
    def _preprocess_points(self, Xpc: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.x_center:
            Xpc = Xpc - Xpc.mean(dim=1, keepdim=True)
        if self.x_row_norm:
            Xpc = F.normalize(Xpc, dim=-1, eps=eps)
        return Xpc

    def encode(self, x: torch.Tensor, active_branches: Optional[List[str]] = None):
        """
        active_branches: branch subset for ablation (e.g., ["xu","xtu"])
        """
        X_row = self._preprocess_points(x)                     # (B,dH,dW)
        X_col = self._preprocess_points(x.transpose(1, 2))      # (B,dW,dH)

        def proj_flat(pr: torch.Tensor, proj_layer: nn.Linear) -> torch.Tensor:
            pr_t = pr.transpose(1, 2)                  # (B, r, N_points)
            pr_proj = torch.relu(proj_layer(pr_t))     # (B, r, proj_dim)
            return pr_proj.reshape(pr_proj.shape[0], -1)  # (B, r*proj_dim)

        # compute 4 probe responses
        pr_xu   = _standardize_per_sample(X_row @ self.probes_xu.weight)
        pr_xxtu = _standardize_per_sample(X_row @ (X_row.transpose(1, 2) @ self.probes_xxtu.weight))
        pr_xtu  = _standardize_per_sample(X_col @ self.probes_xtu.weight)
        pr_xtxu = _standardize_per_sample(X_col @ (X_col.transpose(1, 2) @ self.probes_xtxu.weight))

        # proj4: branch-specific projections
        f_xu   = proj_flat(pr_xu,   self.shared_probe_proj_row)
        f_xxtu = proj_flat(pr_xxtu, self.shared_probe_proj_row2)
        f_xtu  = proj_flat(pr_xtu,  self.shared_probe_proj_col)
        f_xtxu = proj_flat(pr_xtxu, self.shared_probe_proj_col2)

        # ablation / subset masking: keep only selected branches
        keep = set(active_branches) if active_branches is not None else {"xu", "xxtu", "xtu", "xtxu"}
        if "xu" not in keep:   f_xu   = f_xu   * 0.0
        if "xxtu" not in keep: f_xxtu = f_xxtu * 0.0
        if "xtu" not in keep:  f_xtu  = f_xtu  * 0.0
        if "xtxu" not in keep: f_xtxu = f_xtxu * 0.0

        f = torch.cat([f_xu, f_xxtu, f_xtu, f_xtxu], dim=-1)  # (B, 4*r*proj_dim)
        rep = self.concat_all4_encoder(f)                     # (B, rep_dim)

        # return the first active branch's probe response for compatibility
        pr_ret = pr_xu
        if active_branches is not None:
            order = ["xu", "xxtu", "xtu", "xtxu"]
            pr_map = {"xu": pr_xu, "xxtu": pr_xxtu, "xtu": pr_xtu, "xtxu": pr_xtxu}
            for b in order:
                if b in keep:
                    pr_ret = pr_map[b]
                    break
        return pr_ret, rep


class ProbeXClassification(ProbeXCore):
    def __init__(
        self,
        input_shape,
        num_classes: int,
        n_probes: int,
        proj_dim: int,
        rep_dim: int,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__(
            input_shape=input_shape,
            n_probes=n_probes,
            proj_dim=proj_dim,
            rep_dim=rep_dim,
            x_center=x_center,
            x_row_norm=x_row_norm,
        )
        self.classification_head = nn.Linear(self.rep_dim_out, num_classes)

    def forward(self, x, return_probe: bool = False, active_branches: Optional[List[str]] = None):
        probe_responses, representation = self.encode(x, active_branches=active_branches)
        logits = self.classification_head(representation)
        if return_probe:
            return logits, probe_responses, representation
        return logits


class ProbeXZeroshot(ProbeXCore):
    def __init__(
        self,
        input_shape,
        clip_dim: int = 768,
        n_probes: int = 64,
        proj_dim: int = 64,
        rep_dim: int = 256,
        x_center: bool = False,
        x_row_norm: bool = False,
    ):
        super().__init__(
            input_shape=input_shape,
            n_probes=n_probes,
            proj_dim=proj_dim,
            rep_dim=rep_dim,
            x_center=x_center,
            x_row_norm=x_row_norm,
        )
        self.regression_head = nn.Linear(clip_dim, rep_dim)

    def forward(self, x, clip_embedding, return_probe: bool = False, active_branches: Optional[List[str]] = None):
        probe_responses, representation = self.encode(x, active_branches=active_branches)

        clip_z = self.regression_head(clip_embedding)
        clip_z = clip_z / clip_z.norm(dim=-1, keepdim=True)
        representation = representation / representation.norm(dim=-1, keepdim=True)

        logits = representation @ clip_z.T
        if return_probe:
            return logits, probe_responses, representation
        return logits

    def extract_representation(self, x):
        _, representation = self.encode(x, active_branches=None)
        return representation
