"""vLLM general plugin: text-only support for the re-published gemma-4-12B-it.

google/gemma-4-12B-it (transformers >= 5.10) ships as Gemma4UnifiedForConditionalGeneration /
model_type gemma4_unified. vLLM 0.22 has no entry for that name and falls back to its generic
Transformers backend, which fails at load (KeyError: attribute 'embed_scale' already exists).
The unified *text* tower is tensor-for-tensor the gemma4 layout vLLM already implements (same
weight names incl. layer_scalar, attention_k_eq_v, no PLE/MoE; verified against
gemma-4-31B-it), while the vision side is a different small patch embedder. We only generate
text, so map the architecture onto vLLM's native text-only Gemma4ForCausalLM and drop the
vision/audio tensors. Registered through the entry-point plugin group so the spawned engine
process sees it too (a registration in the parent process alone is lost under spawn).
"""
ARCH = "Gemma4UnifiedForConditionalGeneration"


def register() -> None:
    from vllm import ModelRegistry
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCH, "vllm_gemma4_unified.model:Gemma4UnifiedForCausalLM")
    # vLLM's per-architecture config hook forces TRITON_ATTN for gemma4's mixed head dims
    # (sliding 256 / global 512); it is keyed by architecture name, so give the unified name
    # the same hook. Without it vLLM picks FlashInfer and tries a JIT build (no nvcc here).
    try:
        from vllm.model_executor.models import config as _mc
        _mc.MODELS_CONFIG_MAP.setdefault(ARCH, _mc.Gemma4Config)
    except Exception as exc:  # pragma: no cover - keep model registration usable regardless
        print(f"[vllm_gemma4_unified] could not install Gemma4 config hook: {exc!r}")
