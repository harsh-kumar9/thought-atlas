from collections.abc import Iterable

import torch

from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

# unified checkpoint tensors with no counterpart in the text-only model
_SKIP = ("vision_embedder.", "embed_vision.", "embed_audio.", "vision_tower.", "audio_tower.")


class Gemma4UnifiedForCausalLM(Gemma4ForCausalLM):
    """Text-only gemma4_unified: native Gemma4ForCausalLM minus the vision/audio tensors."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            (name, w) for name, w in weights if not any(s in name for s in _SKIP))
