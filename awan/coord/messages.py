"""A2A-inspired negotiation message schema (playbook §2.4), versioned
`awan/neg/v1`. Mirrors the Agent2Agent pattern (Agent Card -> propose ->
counter -> accept), which is now a Linux-Foundation standard (v1.0, 150+ orgs,
JSON-RPC/gRPC bindings, signed Agent Cards) — the market-alignment hook. Every
message's real UTF-8 byte length x8 is the bit count charged to the mission
energy budget via the control channel.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

SCHEMA = "awan/neg/v1"


def _bits(obj) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8")) * 8


@dataclass
class AgentCard:
    agent_id: int
    pos: List[float]
    payload_bits: float
    omega_size: int
    battery_frac: float
    capabilities: List[str] = field(default_factory=lambda: ["aggregate", "relay"])
    schema: str = SCHEMA

    def bits(self):
        return _bits(asdict(self))


@dataclass
class Propose:
    frm: int
    to: int
    offered_role: str            # "sender" | "receiver"
    est_pair_energy_J: float
    est_value: float
    schema: str = SCHEMA
    kind: str = "propose"

    def bits(self):
        return _bits(asdict(self))


@dataclass
class Counter:
    frm: int
    to: int
    offered_role: str
    est_pair_energy_J: float
    schema: str = SCHEMA
    kind: str = "counter"

    def bits(self):
        return _bits(asdict(self))


@dataclass
class Accept:
    pair: List[int]
    role_map: dict               # {"sender": id, "receiver": id}
    schema: str = SCHEMA
    kind: str = "accept"

    def bits(self):
        return _bits(asdict(self))


@dataclass
class Decline:
    frm: int
    to: int
    reason: str
    schema: str = SCHEMA
    kind: str = "decline"

    def bits(self):
        return _bits(asdict(self))


ACTION_KINDS = ("propose", "counter", "accept", "decline")


def validate_action(obj) -> Optional[str]:
    """Return None if `obj` is a schema-valid action, else an error string."""
    if not isinstance(obj, dict):
        return "not a JSON object"
    kind = obj.get("kind")
    if kind not in ACTION_KINDS:
        return f"bad/missing kind {kind!r}"
    req = {"propose": ("to", "offered_role"),
           "counter": ("to", "offered_role"),
           "accept": ("pair",),
           "decline": ("to",)}[kind]
    for k in req:
        if k not in obj:
            return f"missing field {k!r} for {kind}"
    if kind in ("propose", "counter") and obj["offered_role"] not in ("sender", "receiver"):
        return f"bad offered_role {obj.get('offered_role')!r}"
    return None
