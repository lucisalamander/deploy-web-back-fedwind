"""
Unified PEFT (Parameter-Efficient Fine-Tuning) utilities for all LLM backbones.

Supports: LoRA, LoHA, AdaLoRA, PFT (partial fine-tuning), FFT (full fine-tuning).
"""

from peft import LoraConfig, LoHaConfig, AdaLoraConfig, get_peft_model


def apply_peft(model, configs, target_modules):
    """
    Apply the selected PEFT method to an LLM backbone model.

    Args:
        model: The HuggingFace model to wrap.
        configs: SimpleNamespace with peft_method, lora_r, lora_alpha, lora_dropout.
        target_modules: List of module names to apply adapters to (model-specific).

    Returns:
        The (possibly wrapped) model.
    """
    method = getattr(configs, "peft_method", "lora")
    if method in ("pft", "fft"):
        return model

    r = getattr(configs, "lora_r", 8)
    alpha = getattr(configs, "lora_alpha", 16)
    dropout = getattr(configs, "lora_dropout", 0.15)

    if method == "lora":
        peft_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
    elif method == "loha":
        peft_cfg = LoHaConfig(
            r=r,
            alpha=alpha,
            target_modules="all-linear",  # GPT-2 uses Conv1D; "all-linear" handles it transparently
            module_dropout=dropout,
        )
    elif method == "adalora":
        peft_cfg = AdaLoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            total_step=1000,
        )
    else:
        raise ValueError(f"Unknown peft_method '{method}'. Use: lora, loha, adalora, pft, fft")

    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def is_adapter_param(name):
    """Check if a parameter name belongs to a PEFT adapter (LoRA/LoHA/AdaLoRA)."""
    return "lora_" in name or "hada_" in name


def freeze_backbone(model, configs, norm_keywords):
    """
    Freeze backbone parameters, keeping norm layers, embeddings, and adapter params trainable.

    Args:
        model: The model (possibly PEFT-wrapped).
        configs: SimpleNamespace with freeze, pretrain, peft_method.
        norm_keywords: List of substrings that identify norm/embed params to keep trainable.
            E.g. ['ln', 'wpe'] for GPT-2, ['norm', 'embed'] for LLaMA.
    """
    if not (configs.freeze and configs.pretrain):
        return

    peft_method = getattr(configs, "peft_method", "lora")
    for name, param in model.named_parameters():
        keep = False
        # Always keep norm/embed layers trainable
        for kw in norm_keywords:
            if kw in name.lower():
                keep = True
                break
        # Keep adapter params trainable
        if peft_method in ("lora", "loha", "adalora") and is_adapter_param(name):
            keep = True
        param.requires_grad = keep
