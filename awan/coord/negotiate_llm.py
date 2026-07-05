"""Rung C — the LLM-agent negotiation layer (playbook §2.4), the agentic
centerpiece.

Agents exchange A2A-style JSON messages (AgentCard -> Propose -> Counter/Accept/
Decline) to form pairs under partial information. A bounded FSM guarantees
termination and a valid matching: round-robin by id, <=1 Propose per agent per
cycle, recipient must reply within the cycle, <=3 cycles, then leftovers fall
back to distributed greedy. Every message's real byte length is charged to the
mission energy budget (comm_control), and the LLM engine's prompt+completion
tokens are charged as `negotiation_llm` energy at e_tok J/token — the first
protocol-overhead-in-joules accounting for semantic aggregation.

Engines:
  mock     : deterministic scripted policy (statistics workhorse, e_tok = 0)
  local-hf : a small instruct LLM via transformers; JSON-parse w/ one retry,
             then fall back to the mock action for that turn (counted)
"""

import json
import re

import numpy as np

from .control import ControlChannel
from .common import add_leftovers, order_pair, symmetric_weights
from .greedy_dist import _greedy_match
from .messages import (Accept, AgentCard, Decline, Propose, validate_action)

INF = float("inf")


def make_negotiation_coordinator(cfg=None, engine="mock", accept_frac=0.10,
                                 llm=None, stats=None):
    cfg = cfg or {}
    r_comm = cfg.get("R_comm_m", 250.0)
    e_in = cfg.get("e_tok_in_J", 0.15e-3)
    e_out = cfg.get("e_tok_out_J", 0.45e-3)
    ctrl_holder, stats = {}, (stats if stats is not None else {})

    def coordinator(ctx):
        ctrl = ControlChannel(ctx["p"], ledger=ctx["ledger"], cfg=cfg)
        ctrl_holder["c"] = ctrl
        agents, active, w = ctx["agents"], ctx["active"], ctx["w"]
        c = symmetric_weights(active, w, agents=agents, r_comm=r_comm)
        res = negotiate(active, c, w, agents, ctrl, ctx["round"], r_comm,
                        engine, accept_frac, llm, e_in, e_out, ctx["ledger"])
        for k, v in res["stats"].items():
            stats[k] = stats.get(k, 0) + v
        stats.setdefault("transcript", res["transcript"])
        return res["pairs"]

    coordinator.ctrl_holder = ctrl_holder
    coordinator.stats = stats
    return coordinator


def _card(agents, i):
    a = agents[i]
    return AgentCard(agent_id=i, pos=[float(a["pos"][0]), float(a["pos"][1])],
                     payload_bits=float(a["L"]), omega_size=len(a.get("srcs", {i: 0})),
                     battery_frac=float(a.get("battery_frac", 1.0)))


def negotiate(active, c, w, agents, ctrl, rnd, r_comm, engine, accept_frac,
              llm, e_in, e_out, ledger):
    """The bounded negotiation FSM. Returns pairs + transcript + token/msg stats."""
    unmatched = list(active)
    prefs = {i: sorted((cij, j) for (a, b), cij in c.items()
                       for (ii, j) in [(i, b if a == i else a)] if a == i or b == i)
             for i in active}
    pairs, transcript = [], []
    stats = {"schema_valid": 0, "schema_total": 0, "retries": 0,
             "tok_in": 0, "tok_out": 0, "llm_calls": 0, "n_msgs": 0}

    for cycle in range(3):
        if len(unmatched) < 2:
            break
        proposed = set()
        for i in sorted(unmatched):
            if i not in unmatched or i in proposed:
                continue
            cand = [(cij, j) for cij, j in prefs[i] if j in unmatched and j != i]
            if not cand:
                continue
            best_cost, j = cand[0]
            role = "sender" if w.get((i, j), INF) <= w.get((j, i), INF) else "receiver"
            prop = Propose(frm=i, to=j, offered_role=role,
                           est_pair_energy_J=float(best_cost),
                           est_value=float(-best_cost))
            ctrl.send(prop.bits(), _d(agents, i, j, r_comm), rnd=rnd, agent=i, note="Propose")
            transcript.append(("propose", i, j, round(best_cost, 4)))
            stats["n_msgs"] += 1
            # recipient decision
            action = _recipient_action(j, i, best_cost, c, unmatched, engine,
                                       llm, agents, r_comm, stats, transcript)
            if action == "accept":
                acc = Accept(pair=[i, j],
                             role_map={"sender": i if role == "sender" else j,
                                       "receiver": j if role == "sender" else i})
                ctrl.send(acc.bits(), _d(agents, i, j, r_comm), rnd=rnd, agent=j, note="Accept")
                stats["n_msgs"] += 1
                pairs.append((i, j))
                unmatched.remove(i); unmatched.remove(j)
                proposed.add(i); proposed.add(j)
            else:
                dec = Decline(frm=j, to=i, reason="better_option")
                ctrl.send(dec.bits(), _d(agents, i, j, r_comm), rnd=rnd, agent=j, note="Decline")
                stats["n_msgs"] += 1
                proposed.add(i)

    if e_out > 0 and (stats["tok_in"] or stats["tok_out"]):
        e_neg = stats["tok_in"] * e_in + stats["tok_out"] * e_out
        ledger.add(e_neg, "negotiation_llm", rnd=rnd, source="MODELED",
                   note=f"{engine} negotiation tokens")

    # leftovers -> distributed greedy (guarantees a valid matching)
    if len(unmatched) >= 2:
        cc = {(min(i, j), max(i, j)): c[(min(i, j), max(i, j))]
              for i in unmatched for j in unmatched if i < j
              and (min(i, j), max(i, j)) in c}
        for (i, j) in _greedy_match(unmatched, cc, agents, ctrl, rnd, r_comm):
            if j is not None:
                pairs.append((i, j)); unmatched = [x for x in unmatched if x not in (i, j)]

    oriented = [order_pair(i, j, w) for (i, j) in pairs]
    return {"pairs": add_leftovers(oriented, active), "transcript": transcript,
            "stats": stats}


def _recipient_action(j, i, offered_cost, c, unmatched, engine, llm, agents,
                      r_comm, stats, transcript):
    """Recipient j decides accept/decline. mock = scripted; local-hf = LLM."""
    my_best = min((cij for (a, b), cij in c.items()
                   if (a == j or b == j) and (b if a == j else a) in unmatched
                   and (b if a == j else a) != j), default=INF)
    scripted = "accept" if offered_cost <= my_best * (1.0 + 0.10) else "decline"
    if engine == "mock" or llm is None:
        return scripted
    # local-hf: ask the model for a JSON action, parse w/ one retry, fall back
    prompt = _build_prompt(agents, j, i, offered_cost, my_best)
    for attempt in range(2):
        out, tin, tout = llm(prompt)
        stats["tok_in"] += tin; stats["tok_out"] += tout; stats["llm_calls"] += 1
        stats["schema_total"] += 1
        obj = _extract_json(out)
        err = validate_action(obj) if obj is not None else "no json"
        if err is None:
            stats["schema_valid"] += 1
            return "accept" if obj["kind"] == "accept" else "decline"
        if attempt == 0:
            stats["retries"] += 1
    return scripted


def _build_prompt(agents, j, i, offered_cost, my_best):
    cj, ci = _card(agents, j), _card(agents, i)
    return (
        "You are a wireless agent negotiating who to pair with to aggregate data "
        "at minimum energy. Lower est_pair_energy_J is better.\n"
        f"Your card: {json.dumps({'agent_id': cj.agent_id, 'battery_frac': cj.battery_frac})}\n"
        f"Incoming Propose from agent {i}: est_pair_energy_J={offered_cost:.4f}. "
        f"Your best alternative pair energy is {my_best:.4f}.\n"
        'Reply with ONE JSON action ONLY, no prose. To pair, reply '
        f'{{"kind":"accept","pair":[{i},{j}]}}. To refuse, reply '
        f'{{"kind":"decline","to":{i}}}.'
    )


def _extract_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _d(agents, i, j, r_comm):
    return min(float(np.linalg.norm(agents[i]["pos"] - agents[j]["pos"])), r_comm)
