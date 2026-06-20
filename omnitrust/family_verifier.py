"""
omnitrust/family_verifier.py — Collaborative Family Attention Verification Engine.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Class implementing Family Attention mechanisms in PyTorch/NumPy
class FamilyAttentionVerifier:
    """
    Fuses the custom Family Attention mechanism into a fact verification layer.
    Uses learnable attention head identity vectors and lateral representation mixing
    to evaluate factual statement consistency from multiple independent perspective vectors.
    """
    def __init__(self, num_heads: int = 4, embedding_dim: int = 64):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.d_head = embedding_dim // num_heads

        # In-memory mock parameter arrays (NumPy fallbacks)
        self.numpy_id_vecs = [[random.uniform(-1.0, 1.0) for _ in range(self.d_head)] for _ in range(num_heads)]
        self.numpy_lateral_gate = -2.0  # Logit scale for routing gating

        if HAS_TORCH:
            self.torch_module = PyTorchFamilyVerifierModule(num_heads, embedding_dim)
        else:
            self.torch_module = None

    def compute_js_divergence(self, probs: List[List[float]]) -> float:
        """
        Computes the average pairwise Jensen-Shannon Divergence across head probability outputs.
        High divergence = high head diversity (representation coverage).
        Low divergence = heads collapsing into redundant viewpoints.
        """
        h = len(probs)
        if h <= 1:
            return 0.0

        def kl_divergence(p, q):
            return sum(pi * math.log(max(pi, 1e-8) / max(qi, 1e-8)) for pi, qi in zip(p, q) if pi > 0)

        js_sum = 0.0
        pairs = 0
        for i in range(h):
            for j in range(i + 1, h):
                p = probs[i]
                q = probs[j]
                # Mixture distribution
                m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
                js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
                js_sum += js
                pairs += 1

        return js_sum / pairs if pairs > 0 else 0.0

    def verify_statements(self, claims: List[str], evidence: List[str]) -> Dict[str, Any]:
        """
        Simulates running the statements and evidence through the collaborative heads.
        Calculates verification scores, head representation diversity (JS Divergence),
        and stability scores.
        """
        # If PyTorch is available, we can execute the actual neural network checks.
        # Otherwise, we calculate using mathematical formulations via NumPy/pure python.
        num_claims = len(claims)
        if num_claims == 0:
            return {"consistency_score": 1.0, "js_divergence": 0.0, "head_perspectives": []}

        # Normalize identity vectors
        norm_id_vecs = []
        for vec in self.numpy_id_vecs:
            norm = math.sqrt(sum(v*v for v in vec))
            norm_id_vecs.append([v / max(norm, 1e-8) for v in vec])

        # Compute lateral similarity matrix S(H, H)
        sim_matrix = []
        for i in range(self.num_heads):
            row = []
            for j in range(self.num_heads):
                sim = sum(norm_id_vecs[i][k] * norm_id_vecs[j][k] for k in range(self.d_head))
                row.append(sim)
            # Softmax on each row
            exp_row = [math.exp(s) for s in row]
            sum_exp = sum(exp_row)
            row_softmax = [e / sum_exp for e in exp_row]
            sim_matrix.append(row_softmax)

        # Evaluate claim consistency using evidence overlap
        head_perspectives = []
        consistency_scores = []
        
        # Simulating distinct head focus behaviors
        head_focus_templates = [
            {"name": "Semantic Head", "weights": [0.6, 0.2, 0.2]},
            {"name": "Temporal/Metric Head", "weights": [0.2, 0.6, 0.2]},
            {"name": "Named Entity Head", "weights": [0.1, 0.3, 0.6]},
            {"name": "Context Head", "weights": [0.3, 0.3, 0.4]}
        ]

        for h_idx in range(self.num_heads):
            head_probs = []
            focus = head_focus_templates[h_idx % len(head_focus_templates)]
            
            # Simple heuristic matching score for the claim against evidence
            for claim in claims:
                words = set(claim.lower().split())
                ev_words = set(" ".join(evidence).lower().split())
                
                # Check entity and metric markers
                digits_overlap = len([w for w in words if w.isdigit() and w in ev_words])
                negation_overlap = len(words.intersection({"not", "never", "no"}))
                base_overlap = len(words.intersection(ev_words)) / max(len(words), 1)

                # Each head weighs features differently
                w = focus["weights"]
                match_val = (w[0] * base_overlap) + (w[1] * min(digits_overlap, 1.0)) - (w[2] * negation_overlap)
                prob_supported = 1.0 / (1.0 + math.exp(-10.0 * (match_val - 0.2)))
                
                # Probability distribution: [Supported, Uncertain, Unsupported]
                p_supported = prob_supported
                p_unsupported = 0.8 * (1.0 - p_supported) if negation_overlap > 0 else 0.1 * (1.0 - p_supported)
                p_uncertain = 1.0 - p_supported - p_unsupported
                
                head_probs.append([p_supported, p_uncertain, p_unsupported])

            # Average probability distribution of this head across claims
            avg_probs = [sum(head_probs[c][idx] for c in range(num_claims)) / num_claims for idx in range(3)]
            head_perspectives.append(avg_probs)
            
            # Use support probability as base consistency
            consistency_scores.append(avg_probs[0])

        # Compute lateral mix update using sim_matrix
        mixed_scores = []
        lateral_gate_val = 1.0 / (1.0 + math.exp(-self.numpy_lateral_gate))
        for i in range(self.num_heads):
            # mixed_i = sum_j E_ij * score_j
            mixed = sum(sim_matrix[i][j] * consistency_scores[j] for j in range(self.num_heads))
            # gated update
            mixed_scores.append(consistency_scores[i] + lateral_gate_val * mixed)

        # Average verification consistency
        consistency_score = sum(mixed_scores) / self.num_heads
        # Bound between 0 and 1
        consistency_score = max(0.0, min(1.0, consistency_score))

        # Compute JS Divergence of head perspective distributions
        js_div = self.compute_js_divergence(head_perspectives)

        return {
            "consistency_score": round(consistency_score, 4),
            "js_divergence": round(js_div, 4),
            "head_perspectives": [
                {
                    "head_id": idx,
                    "name": head_focus_templates[idx % len(head_focus_templates)]["name"],
                    "supported": round(p[0], 4),
                    "uncertain": round(p[1], 4),
                    "unsupported": round(p[2], 4)
                }
                for idx, p in enumerate(head_perspectives)
            ],
            "similarity_matrix": [[round(val, 4) for val in row] for row in sim_matrix]
        }


# --- PyTorch Module (only instantiated if torch is installed) ---
if HAS_TORCH:
    class PyTorchFamilyVerifierModule(nn.Module):
        def __init__(self, num_heads: int, embedding_dim: int):
            super().__init__()
            self.num_heads = num_heads
            self.d_head = embedding_dim // num_heads
            self.id_vecs = nn.Parameter(torch.randn(num_heads, self.d_head))
            self.lateral_gate = nn.Parameter(torch.tensor(-2.0))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x shape: (B, N, embedding_dim)
            B, N, D = x.shape
            H = self.num_heads
            d_h = self.d_head

            # Normalize identity vectors
            id_n = F.normalize(self.id_vecs, p=2, dim=-1) # (H, d_head)
            sim = id_n @ id_n.T # (H, H)
            e_mat = F.softmax(sim, dim=-1) # (H, H)

            # Reshape inputs into heads
            h = x.view(B, N, H, d_h) # (B, N, H, d_head)

            # Lateral mix multiplication
            mixed = torch.matmul(e_mat, h) # (B, N, H, d_head)
            gate_val = torch.sigmoid(self.lateral_gate)
            h_updated = h + gate_val * mixed

            # Compute JS divergence loss surrogate (mean-pool and soft distribution)
            h_pooled = h_updated.mean(dim=1) # (B, H, d_head)
            p = F.softmax(h_pooled, dim=-1) # (B, H, d_head)

            # Pairwise JS Divergence calculation
            p_i = p.unsqueeze(2) # (B, H, 1, d_head)
            p_j = p.unsqueeze(1) # (B, 1, H, d_head)
            M = 0.5 * (p_i + p_j) # (B, H, H, d_head)

            kl_i = (p_i * (p_i.clamp(min=1e-8).log() - M.clamp(min=1e-8).log())).sum(dim=-1) # (B, H, H)
            kl_j = (p_j * (p_j.clamp(min=1e-8).log() - M.clamp(min=1e-8).log())).sum(dim=-1) # (B, H, H)
            js_mat = 0.5 * (kl_i + kl_j) # (B, H, H)

            mask = (~torch.eye(H, dtype=torch.bool, device=x.device))
            # average JS per head vs other heads
            js_divs = js_mat[:, mask].view(B, H, H - 1).mean(dim=-1).mean(dim=0) # (H,)

            return h_updated.reshape(B, N, D), js_divs
