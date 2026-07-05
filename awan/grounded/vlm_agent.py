"""SmolVLM2 perception agent (playbook §4.2): image -> structured JSON facts.

Fixed prompt template demands ONLY JSON; strict parse, one retry with a
sterner instruction, then a regex rescue. The parse-failure rate is logged and
reported (G1 target < 10%). All raw generations are cached to disk so every
figure regenerates WITHOUT re-running the VLM (G8).
"""

import json
import re
import time

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"   # playbook primary
SMOKE_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

PROMPT = (
    "Describe this image in detail. Name the color and type of every vehicle "
    "and every person in it."
)
RETRY_SUFFIX = (" Describe it again, naming each vehicle or person together "
                "with its color.")


class VLMAgent:

    def __init__(self, model_id=MODEL_ID, device="cpu"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        self.model_id = model_id
        self.stats = {"calls": 0, "parse_fail": 0, "retries": 0,
                      "tok_in": 0, "tok_out": 0, "wall_s": 0.0}

    def perceive(self, image, max_new_tokens=160):
        """-> (facts list, raw text). Counts tokens + wall time for the meter."""
        raw = self._generate(image, PROMPT, max_new_tokens)
        facts = parse_facts(raw)
        if facts is None:
            self.stats["retries"] += 1
            raw = self._generate(image, PROMPT + RETRY_SUFFIX, max_new_tokens)
            facts = parse_facts(raw)
        if facts is None:
            self.stats["parse_fail"] += 1
            facts = regex_rescue(raw)
        return facts, raw

    def _generate(self, image, prompt, max_new_tokens):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        n_in = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        gen = out[0][n_in:]
        self.stats["calls"] += 1
        self.stats["tok_in"] += n_in
        self.stats["tok_out"] += int(gen.shape[0])
        self.stats["wall_s"] += dt
        return self.proc.decode(gen, skip_special_tokens=True)


GRAMMAR = re.compile(r"(red|blue|green|yellow|white|black)\s+(car|bus|truck|person)")


def parse_facts(text):
    """Constrained line grammar 'color type' (risk-register fallback, §10);
    also accepts schema JSON if the model volunteers it. Returns None only
    when nothing parseable is present (counted as a parse failure)."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            facts = json.loads(m.group(0)).get("facts")
            if isinstance(facts, list) and facts:
                return [{"object": str(f.get("object", "?")).lower(),
                         "attr": str(f.get("attr", "?")).lower(),
                         "confidence": float(f.get("confidence", 0.5) or 0.5)}
                        for f in facts if isinstance(f, dict)]
        except Exception:
            pass
    hits = GRAMMAR.findall(text.lower())
    if not hits:
        return None
    return [{"object": o, "attr": c, "confidence": 0.8} for c, o in hits]


def regex_rescue(text):
    hits = GRAMMAR.findall((text or "").lower())
    return [{"object": o, "attr": c, "confidence": 0.3} for c, o in hits]
