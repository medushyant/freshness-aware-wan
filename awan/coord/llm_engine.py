"""Local small-LLM engine for Rung-C negotiation (playbook §2.4, engine
`local-hf`). A real instruct model via transformers, greedy-decoded to a short
JSON action. Counts prompt/completion tokens for the energy ledger. Lazily
loaded and fully guarded: if transformers/torch or the weights are unavailable,
`load_llm` returns None and the caller uses the mock engine (statistics are
mock-based by design).
"""

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small, Apache-2.0, CPU-runnable


def load_llm(model_id=DEFAULT_MODEL, max_new_tokens=24):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
    except Exception:
        return None

    @torch.no_grad()
    def llm(prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt")
        n_in = int(enc["input_ids"].shape[1])
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        n_out = int(gen.shape[0])
        return tok.decode(gen, skip_special_tokens=True), n_in, n_out

    llm.model_id = model_id
    return llm
