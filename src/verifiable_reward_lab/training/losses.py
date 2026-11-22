"""Training objectives built from raw tensors.

Implements token-level cross-entropy for autoregressive language modelling,
with shape annotations at every step.

Kept out of the model on purpose. `DecoderTransformer.forward` already
returns `(logits, caches)`; threading a `targets=None` argument through it
would make the arity of the return value depend on the arity of the call.
The model produces logits. Turning logits into a differentiable scalar is
a separate job, and it lives here.

Reference: "Cross-Entropy for Language Models"
    https://www.vitorsousa.com/bits/cross-entropy-language-models
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

IGNORE_INDEX = -100
"""The label cross-entropy skips. Lives here, with the objective that
honours it, rather than with the code that happens to write it."""

# =========================================================================
# Token-level cross-entropy
# =========================================================================


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over every position in the batch.

    Every one of the B*T positions is an independent next-token
    prediction, which is exactly why a window of T tokens is T training
    examples and not one. The batch and time axes carry no meaning for
    the loss, so they are flattened away.

    Sanity check: an untrained model must score ln(V) — exactly as
    confused as a uniform guess over the vocabulary. If that number is
    wrong, every loss curve downstream is measuring the wrong thing.

    Args:
        logits:  (B, T, V) float. Raw scores, NOT softmaxed —
            `F.cross_entropy` applies log_softmax internally, and doing
            it twice silently flattens the distribution instead of
            raising.
        targets: (B, T) long. The next-token ids from `get_batch`.

    Returns:
        (), a 0-dim tensor carrying grad. Scalar, because `.backward()`
        needs one.

    Raises:
        ValueError: If the leading dims disagree, or targets is not long.
    """
    # =========================================
    # 1. Validate — name both shapes
    # =========================================
    if logits.shape[:2] != targets.shape:
        msg = (
            f"leading dims must match: logits {tuple(logits.shape)} "
            f"gives (B, T) = {tuple(logits.shape[:2])}, "
            f"targets is {tuple(targets.shape)}"
        )
        raise ValueError(msg)

    if targets.dtype != torch.long:
        msg = (
            f"targets must be torch.long, got {targets.dtype}. "
            f"Check the dtype coming out of get_batch."
        )
        raise ValueError(msg)

    # =========================================
    # 2. Flatten to the (N, C) / (N,) form F.cross_entropy wants
    # =========================================
    vocab_size = logits.size(-1)

    # reshape, not view. view() requires contiguous memory and raises if
    # the logits came out of an op that left non-contiguous strides —
    # which depends on the lm head. reshape falls back to a copy.
    # (B, T, V) -> (B*T, V)
    flat_logits = logits.reshape(-1, vocab_size)
    # (B, T) -> (B*T,)
    flat_targets = targets.reshape(-1)

    # =========================================
    # 3. Reduce — "mean" written explicitly
    # =========================================
    # The difference between mean and sum is a factor of B*T: the loss
    # still falls and the curve still looks like learning, but the
    # effective learning rate is B*T times what you think it is.
    # (B*T, V), (B*T,) -> ()
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean")


# =========================================================================
# Causal LM cross-entropy over a masked batch
# =========================================================================


def causal_lm_loss(
    logits: Tensor, labels: Tensor, *, ignore_index: int = IGNORE_INDEX
) -> Tensor:
    """Cross-entropy over the labelled positions of a causal batch.

    Differs from `cross_entropy_loss` in who does the shifting. There,
    `get_batch` hands over targets already shifted, and the loss compares
    them position for position. Here the labels are aligned with the input
    ids — label[t] is the token AT t, not the one after it — because that
    is the only alignment in which "mask the prompt" is expressible. The
    shift therefore happens here, once, in the open.

    Hugging Face models shift internally when handed `labels`. This does
    not use that path: an off-by-one that lives inside a dependency is one
    that cannot be unit-tested, and it changes between releases.

    Args:
        logits: (B, T, V) float. Raw scores, NOT softmaxed.
        labels: (B, T) long, aligned with the inputs. Positions the loss
            must skip carry `ignore_index`.
        ignore_index: The label value to skip.

    Returns:
        (), a 0-dim tensor carrying grad. The mean over labelled positions
        only — padding and prompt do not dilute it.

    Raises:
        ValueError: If the leading dims disagree, labels is not long, or
            the batch has no labelled position at all.
    """
    if logits.shape[:2] != labels.shape:
        msg = (
            f"leading dims must match: logits {tuple(logits.shape)} "
            f"gives (B, T) = {tuple(logits.shape[:2])}, "
            f"labels is {tuple(labels.shape)}"
        )
        raise ValueError(msg)

    if labels.dtype != torch.long:
        msg = f"labels must be torch.long, got {labels.dtype}"
        raise ValueError(msg)

    # position t predicts the token at t+1, so the last position predicts
    # nothing. the shift is done on the LABELS, not on the logits: slicing
    # logits[:, :-1, :] leaves a non-contiguous tensor, and flattening it
    # for cross_entropy then copies the whole thing. at Qwen's vocabulary
    # that copy is over 2 GB per microbatch, for a tensor that is thrown
    # away immediately. shifting the labels and marking the final position
    # ignore_index keeps exactly the same terms in the mean and costs a
    # tensor of ints.
    # (B, T) -> (B, T)
    tail = torch.full(
        (labels.size(0), 1), ignore_index, dtype=labels.dtype, device=labels.device
    )
    shifted_labels = torch.cat([labels[:, 1:], tail], dim=1)

    # an all-masked batch would make F.cross_entropy return nan, which
    # backpropagates into every parameter and is only noticed later, in a
    # loss curve that went flat for no visible reason
    if int((shifted_labels != ignore_index).sum()) == 0:
        msg = "batch has no labelled position; every label is ignore_index"
        raise ValueError(msg)

    vocab_size = logits.size(-1)
    # the model's own logits are contiguous, so this reshape is a view
    # (B, T, V) -> (B*T, V)
    flat_logits = logits.reshape(-1, vocab_size)
    # (B, T) -> (B*T,)
    flat_labels = shifted_labels.reshape(-1)

    # reduction="mean" with ignore_index divides by the number of KEPT
    # positions, not by B*T. that is what makes a batch of long answers
    # and a batch of short ones comparable.
    return F.cross_entropy(
        flat_logits, flat_labels, ignore_index=ignore_index, reduction="mean"
    )
