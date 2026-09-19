"""Vendored MiniMax-H3 model code.

See ../UPSTREAM.md for provenance and the rules for modifying these files. This one is *ahead of
the pin*: MiniMax-H3 does not exist in the pinned diffusers 0.39.0, so the file was taken from a
`main` checkout. Every upstream symbol it imports was checked to exist in 0.39.0, which is what
makes that safe without moving the pin.
"""

from .transformer_minimax_h3 import MiniMaxH3Transformer3DModel

__all__ = ["MiniMaxH3Transformer3DModel"]
