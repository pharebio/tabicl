from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProjectionEncoder(nn.Module):
    """Map a fixed-width row straight to the ICL stage's input, with no attention.

    This is the terminal ablation of TabICL's first two stages (see
    ``docs/design/no_col_row_attention.md``). ``ColEmbedding`` and ``RowInteraction``
    are not replaced by cheaper attention -- they are deleted, and a row vector is
    projected to ``embed_dim * num_cls`` by an MLP. It is the architecture that
    molecular in-context-learning models assume when they feed a fixed-length encoder
    output straight into an ICL transformer.

    It removes four things at once, and a drop in accuracy cannot be attributed to any
    one of them without the intermediate rungs:

    1. Context conditioning -- a cell's embedding no longer depends on the distribution
       of its column across the context rows.
    2. Target awareness -- ``ColEmbedding`` runs with ``col_target_aware=True``; here
       ``y`` reaches the model only where ``ICLearning`` adds it to the context rows.
    3. Feature interaction -- no attention across the features of a row.
    4. Column-order invariance -- each input position gets its own weights, so feature
       shuffling is no longer a symmetry and no longer generates ensemble members.

    The output matches what ``RowInteraction`` produces, not merely in shape: that
    module applies ``LayerNorm(embed_dim)`` to each of its ``num_cls`` CLS-token outputs
    before flattening them, so each ``embed_dim``-wide chunk of the result is
    normalized. ``out_ln`` here does the same, which is what lets a pretrained
    ``ICLearning`` read the result.

    Rows are independent, so inference streams them in chunks of ``chunk_size``; there
    is no ``InferenceManager``, because the output is ``(B, T, embed_dim * num_cls)``
    and nothing larger is ever materialized.

    Parameters
    ----------
    max_features : int
        Fixed input width. Inputs with fewer columns are zero-padded on the right;
        inputs with more are refused rather than truncated.

    embed_dim : int
        Per-CLS-token width, matching the row stage being replaced.

    num_cls : int
        Number of CLS-token slots, matching the row stage being replaced. The output
        width is ``embed_dim * num_cls``.

    ff_factor : int, default=2
        Hidden width of the MLP, as a multiple of the output width.

    dropout : float, default=0.0
        Dropout probability inside the MLP.

    activation : str or unary callable, default="gelu"
        Activation used in the MLP.

    bias_free_ln : bool, default=False
        If True, removes bias from the output LayerNorm.

    chunk_size : int, default=65536
        Rows per forward chunk at inference.
    """

    def __init__(
        self,
        max_features: int,
        embed_dim: int,
        num_cls: int,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        bias_free_ln: bool = False,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.max_features = max_features
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.chunk_size = chunk_size

        d_model = embed_dim * num_cls
        self.d_model = d_model

        if isinstance(activation, str):
            act = {"gelu": nn.GELU, "relu": nn.ReLU}.get(activation)
            if act is None:
                raise ValueError(f"Unsupported activation '{activation}'. Use 'gelu', 'relu', or a callable.")
            act = act()
        else:
            act = activation

        self.mlp = nn.Sequential(
            nn.Linear(max_features, d_model * ff_factor),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_factor, d_model),
        )
        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln)

    def _project(self, X: Tensor) -> Tensor:
        """Project one chunk of rows. ``X`` is (B, t, H); returns (B, t, num_cls*embed_dim)."""
        H = X.shape[-1]
        if H > self.max_features:
            raise ValueError(
                f"Input has {H} features but this encoder was built for max_features="
                f"{self.max_features}. Truncating would silently drop {H - self.max_features} "
                f"columns, so it is refused: rebuild the model with proj_max_features >= {H}. "
                f"Unlike the attention stages it replaces, this encoder has a fixed input width."
            )
        if H < self.max_features:
            X = F.pad(X, (0, self.max_features - H), value=0.0)

        out = self.mlp(X)  # (B, t, d_model)
        out = out.unflatten(-1, (self.num_cls, self.embed_dim))  # (B, t, C, E)
        return self.out_ln(out).flatten(-2)  # (B, t, C*E)

    def forward(self, X: Tensor, y_train: Optional[Tensor] = None, **unused) -> Tensor:
        """Project rows to the ICL stage's input.

        Parameters
        ----------
        X : Tensor
            Input of shape (B, T, H). Zero-padded to ``max_features``; ``H`` greater
            than ``max_features`` raises.

        y_train : Optional[Tensor], default=None
            Accepted for signature parity with the stages this replaces, and ignored.
            The encoder is deliberately target-agnostic; see the class docstring.

        **unused
            Swallows the keyword arguments the replaced stages take (``d``,
            ``embed_with_test``, ``feature_shuffles``, ``mgr_config``) so the two paths
            are interchangeable at the call site. ``feature_shuffles`` is a no-op here
            by construction, since this encoder is not column-order invariant.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, embed_dim * num_cls).
        """
        if self.training or X.shape[1] <= self.chunk_size:
            return self._project(X)

        return torch.cat(
            [self._project(X[:, i : i + self.chunk_size]) for i in range(0, X.shape[1], self.chunk_size)],
            dim=1,
        )
