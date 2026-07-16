"""World-model encoder/decoder package (roadmap M3, design §4.2-4.3).

An encoder that compresses a tokenized combat state into a normalized latent ``z`` (set transformer +
SimNorm) and a symbolic decoder that reconstructs the structured state from ``z`` alone. The decoder is
the training signal, the anti-collapse anchor, and the debugger. See :mod:`lts2_agent.train_encdec` for
the trainer and :mod:`lts2_agent.eval_encdec` for the full-split report card.
"""

from __future__ import annotations

from .model import WorldModelAE, compute_losses, featurize, load_checkpoint, save_checkpoint

__all__ = ["WorldModelAE", "compute_losses", "featurize", "load_checkpoint", "save_checkpoint"]
