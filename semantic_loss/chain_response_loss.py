"""Loss fuzzy Direct-Soft per Chain Response(A, B)."""

import torch
from torch.nn import functional as F

from semantic_loss.response_loss import ResponseSoftLoss


class ChainResponseSoftLoss(ResponseSoftLoss):
    """Ogni A richiede B nella posizione immediatamente successiva."""

    def _target_after(self, mass_b, prefix_lengths, prefix_width):
        # Zero dopo l'ultima posizione: una A finale non può avere una B dopo.
        padded_mass_b = F.pad(mass_b, (0, 1))
        next_b = padded_mass_b[:, 1:]

        # Dopo l'ultimo evento reale del prefisso salta il padding e leggi
        # la prima posizione del suffisso (oppure zero se il suffisso è vuoto).
        positions = torch.arange(mass_b.size(1), device=mass_b.device)
        last_prefix = positions == prefix_lengths[:, None] - 1
        first_suffix_b = padded_mass_b[:, prefix_width:prefix_width + 1]
        next_b = torch.where(last_prefix, first_suffix_b, next_b)

        if self.verbose:
            print("-" * 20)
            print("Massa B nella posizione immediatamente successiva")
            print(next_b)
        return next_b
