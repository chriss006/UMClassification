import torch
import torch.nn as nn
import torch.nn.functional as F

NAEVI, IMT, UM = 0, 1, 2   # the only diagnoses we actually have as labels
NUM_THRESHOLDS = 3          # K-1 for K=4 ordinal MOLES buckets (0,1,2,3)


class MolesScoreHead(nn.Module):
    """5 per-criterion sub-score heads (clamped to [0, 2]) that sum into a
    single latent severity score, plus 3 learned ordinal thresholds along
    that same score axis.
    """

    def __init__(self, feat_dim: int, num_criteria: int = 5, temperature: float = 1.0):
        super().__init__()
        self.criteria_proj = nn.Linear(feat_dim, num_criteria)
        # Thresholds spread across the plausible 0-10 sum range; they're
        self.thresholds = nn.Parameter(torch.tensor([1.0, 3.0, 6.0]))
        self.temperature = temperature

    def forward(self, feats: torch.Tensor):
        raw = self.criteria_proj(feats)                  # [B, 5]
        sub_scores = 2.0 * torch.sigmoid(raw)             # each in [0, 2]
        total_score = sub_scores.sum(dim=-1)               # s_hat, in [0, 10]

        # tau_k logit: how far above threshold k is s_hat (in logit units).
        threshold_logits = (
            total_score.unsqueeze(-1) - self.thresholds.unsqueeze(0)
        ) / self.temperature                                # [B, 3]

        return sub_scores, total_score, threshold_logits


def extended_binary_targets(y: torch.Tensor) -> torch.Tensor:
    """y in {NAEVI, IMT, UM} -> [B, 3] float target matrix for (tau_1, tau_2, tau_3).

    tau_2's column is left at 0 for IMT rows; the caller must combine this
    with `threshold_mask` so that column is excluded from the loss rather
    than treated as a real 0 label.
    """
    B = y.shape[0]
    targets = torch.zeros(B, NUM_THRESHOLDS, device=y.device)

    is_um = y == UM
    is_imt = y == IMT
    # naevi (y==NAEVI): all thresholds 0, already the default.

    # tau_1 ("at least class 1"): true for IMT and UM.
    targets[:, 0] = (is_imt | is_um).float()
    # tau_2 ("at least class 2"): true only for UM; unknown/masked for IMT.
    targets[:, 1] = is_um.float()
    # tau_3 ("at least class 3"): true only for UM.
    targets[:, 2] = is_um.float()

    return targets


def threshold_mask(y: torch.Tensor) -> torch.Tensor:
    """[B, 3] mask, 1 = supervised, 0 = excluded from the loss.

    Only tau_2 is ambiguous, and only for IMT rows (naevi and UM pin it down
    unambiguously as 0 and 1 respectively).
    """
    B = y.shape[0]
    mask = torch.ones(B, NUM_THRESHOLDS, device=y.device)
    mask[:, 1] = (y != IMT).float()
    return mask


def coral_loss(threshold_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Masked BCE over the 3 rank-consistent thresholds. Returns the mean
    over only the *unmasked* (label-known) entries.
    """
    targets = extended_binary_targets(y)
    mask = threshold_mask(y)

    per_entry = F.binary_cross_entropy_with_logits(
        threshold_logits, targets, reduction="none"
    )
    return (per_entry * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def decode_ordinal(threshold_logits: torch.Tensor) -> torch.Tensor:
    """Predicted MOLES bucket (0-3) = number of thresholds exceeded."""
    return (torch.sigmoid(threshold_logits) > 0.5).sum(dim=-1)


if __name__ == "__main__":
    # Smoke test on synthetic data.
    torch.manual_seed(0)
    B, feat_dim = 16, 32
    feats = torch.randn(B, feat_dim)
    y = torch.randint(0, 3, (B,))  # NAEVI / IMT / UM labels only

    head = MolesScoreHead(feat_dim)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    for step in range(200):
        sub_scores, s_hat, logits = head(feats)
        loss = coral_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("final loss:", loss.item())
    print("s_hat range:", s_hat.min().item(), s_hat.max().item())
    print("per-criterion sub-score std across heads (collapse check):",
          sub_scores.std(dim=0))
    pred = decode_ordinal(logits)
    print("pred buckets:", pred.tolist())
    print("true labels :", y.tolist())
