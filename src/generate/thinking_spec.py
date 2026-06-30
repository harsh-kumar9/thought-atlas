"""src/generate/thinking_spec.py — per-model thinking configuration + output normalization.

Different reasoning models open/close their think block with different delimiters and enable
thinking differently. Rather than teach the whole downstream pipeline (parse.py, the vendored
ThinkARM segmenter) about every model's delimiters, we NORMALIZE each model's raw generation to
the canonical <think>...</think> form AT CAPTURE TIME (same strategy as the byte-BPE repair).
Everything downstream stays delimiter-agnostic and unchanged.

Specs are keyed by a `thinking_style` string set per model in config (gen_models.<name>.thinking_style).

  r1        : DeepSeek-R1-Distill — chat template opens <think>; model emits </think>. Canonical already.
  gemma4    : Gemma 4 — enable via "<|think|>" at START of system prompt; output wraps reasoning in
              "<|channel>thought\n ... <channel|>" then the answer. Normalize those to <think>/</think>.
  qwen35    : Qwen3.5 — thinks by DEFAULT (no soft switch); emits <think>...</think> like Qwen3.
              Disable via chat_template_kwargs={"enable_thinking": False}. Canonical already.
  none      : non-reasoning control (Llama-3.1-Instruct). No think block.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThinkingSpec:
    style: str
    enable_thinking: bool = True
    # how to switch thinking on/off:
    system_prefix: str = ""                 # text prepended to system prompt to ENABLE (gemma4: "<|think|>")
    template_kwargs: dict = field(default_factory=dict)  # passed to apply_chat_template (qwen: enable_thinking)
    needs_system_role: bool = False         # gemma4 enables via system prompt, so a system turn must exist
    # recommended sampling (model cards)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    extra_sampling: dict = field(default_factory=dict)  # e.g. presence_penalty for qwen
    # raw-output delimiters for the think block (for normalization to canonical <think></think>)
    open_delims: tuple = ()
    close_delims: tuple = ()


SPECS = {
    "r1": ThinkingSpec(
        style="r1", enable_thinking=True,
        temperature=0.6, top_p=0.95, top_k=40,
        open_delims=("<think>",), close_delims=("</think>",)),
    "gemma4": ThinkingSpec(
        style="gemma4", enable_thinking=True,
        system_prefix="<|think|>", needs_system_role=True,
        template_kwargs={"enable_thinking": True},
        temperature=1.0, top_p=0.95, top_k=64,
        # observed channel wrapper; we accept a few variants defensively
        open_delims=("<|channel>thought\n", "<|channel>thought", "<channel>thought\n"),
        close_delims=("<channel|>", "</channel>", "<|channel>")),
    "qwen35": ThinkingSpec(
        style="qwen35", enable_thinking=True,
        template_kwargs={"enable_thinking": True},
        temperature=1.0, top_p=0.95, top_k=20,
        extra_sampling={"presence_penalty": 1.5},
        open_delims=("<think>",), close_delims=("</think>",)),
    "none": ThinkingSpec(
        style="none", enable_thinking=False,
        temperature=0.7, top_p=0.9, top_k=50,
        open_delims=(), close_delims=()),
}


def get_spec(style: str) -> ThinkingSpec:
    if style not in SPECS:
        raise KeyError(f"unknown thinking_style '{style}'; known: {list(SPECS)}")
    return SPECS[style]


def normalize_to_canonical(raw: str, spec: ThinkingSpec) -> str:
    """Rewrite a model's raw think-block delimiters to canonical <think>...</think> so the
    downstream parser/segmenter (which split on </think>) work unchanged. Idempotent: if the text
    is already canonical (r1/qwen) it is returned untouched. Non-reasoning -> untouched."""
    if spec.style in ("r1", "qwen35", "none"):
        return raw                                   # already canonical or no think block
    if not raw:
        return raw
    text = raw
    # replace the FIRST open delimiter occurrence with <think>
    for od in spec.open_delims:
        if od in text:
            text = text.replace(od, "<think>", 1)
            break
    else:
        # model didn't emit a recognizable open delim. If a close delim exists, assume the
        # whole prefix up to it is the thought; prepend <think>. Else leave as-is (no think block).
        if any(cd in text for cd in spec.close_delims):
            text = "<think>" + text
    # replace the FIRST close delimiter with </think>
    for cd in spec.close_delims:
        if cd in text:
            text = text.replace(cd, "</think>", 1)
            break
    return text


def build_messages(user_prompt: str, spec: ThinkingSpec) -> list[dict]:
    """Construct the chat messages, injecting the system prefix that enables thinking if required."""
    msgs = []
    if spec.needs_system_role and spec.enable_thinking and spec.system_prefix:
        msgs.append({"role": "system", "content": spec.system_prefix + "You are a helpful assistant."})
    elif spec.needs_system_role:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def template_kwargs(spec: ThinkingSpec) -> dict:
    return dict(spec.template_kwargs)