"""Pairwise inner-level solver (paper Sec. IV-A) + the Direction-1 version.

Paper mode  : exactly the published problem (P1). Receiver pinned at
              eta_req, sender's eta found by 1-D search, motion by SLSQP.
              Reproduces the eta_j* = eta_req conclusion.
Fidelity mode: our reformulation. Both compression ratios are free,
              the objective gains (a) a distortion penalty lam * dD,
              (b) the future-transport term Phi *inside* the problem
              (the paper only applies Phi after optimizing, see audit L19),
              and the link can use Wyner-Ziv side information (audit L18).

I solve the blocks with scipy instead of CVX: same optimization problems,
lighter dependency stack. The motion block keeps the paper's structure
(positions + motion times + transmit time, smooth constraints).
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .model import (P, chan_gain, rate, mob_energy, comp_load,
                    comp_energy_time, comm_energy, ptx_for)

INF = float("inf")
EPS = 1e-2


def _payload_out(eta_i, L_i, rho_pair, use_wz, p):
    """Bits actually sent over the air. WZ: skip what the receiver knows."""
    out = eta_i * L_i
    if use_wz:
        out *= max(0.0, 1.0 - p["omega_wz"] * rho_pair)
    return out


def _L_next(eta_i, eta_j, L_i, L_j, rho_pair, subadditive):
    """Receiver's payload after fusion. Paper Eq. (2) vs our Eq. (6)."""
    add = eta_j * L_j + eta_i * L_i
    if subadditive:
        add -= rho_pair * min(eta_i * L_i, eta_j * L_j)
    return add


def _resource_step(st, p, opt):
    """Pick compression ratio(s) and transmit time for fixed motion plan."""
    h = chan_gain(st["pe_i"], st["pe_j"], p)
    Rmax = rate(p["Pmax"], h, p)
    Tm_i, Tm_j = st["Tmob_i"], st["Tmob_j"]
    L_i, L_j = opt["L_i"], opt["L_j"]
    lam = opt.get("lam", 0.0)
    fid = opt.get("fidelity", False)

    def proc_time(L, rho, eta):
        return comp_energy_time(L, rho, eta, p)[1]

    def score(eta_i, eta_j):
        Ec_i, Tc_i = comp_energy_time(L_i, opt["rho_i"], eta_i, p)
        Ec_j, Tc_j = comp_energy_time(L_j, opt["rho_j"], eta_j, p)
        t1 = min(p["Tmax"] - max(Tm_i, Tc_i), p["Tmax"] - max(Tm_j, Tc_j))
        if t1 <= EPS:
            return INF, None
        Lout = _payload_out(eta_i, L_i, opt["rho_pair"], opt["use_wz"], p)
        if Lout / Rmax > t1:          # can't make the deadline at full power
            return INF, None
        Ecm = comm_energy(Lout, h, t1, p)
        J = Ec_i + Ec_j + Ecm
        extra = {"t1": t1, "Ecomm": Ecm, "Ec_i": Ec_i, "Ec_j": Ec_j,
                 "Lout": Lout, "h": h}
        if opt.get("phi_in_loop") and not fid:
            Lnx = _L_next(eta_i, eta_j, L_i, L_j, opt["rho_pair"], False)
            J += p["zeta"] * np.linalg.norm(st["pe_j"] - opt["centroid"]) ** p["delta"] * Lnx
        if fid:
            # distortion increase, weighted by each side's share of source mass
            w_i = opt["mass_i"] / (opt["mass_i"] + opt["mass_j"])
            dD = p["a_D"] * (w_i * np.log(1 / eta_i) + (1 - w_i) * np.log(1 / eta_j))
            Lnx = _L_next(eta_i, eta_j, L_i, L_j, opt["rho_pair"], True)
            phi = p["zeta"] * np.linalg.norm(st["pe_j"] - opt["centroid"]) ** p["delta"] * Lnx
            J += lam * dD + phi
            extra["dD"] = dD
        return J, extra

    lo_i, lo_j = opt["eta_lo_i"], opt["eta_lo_j"]
    cap_j = opt["eta_cap_j"]

    if not fid:
        # ---- paper: eta_j sits on its cap (their Lemma), eta_i by 1-D search
        eta_j = cap_j
        res = minimize_scalar(lambda e: score(e, eta_j)[0],
                              bounds=(lo_i, 1.0), method="bounded",
                              options={"xatol": 1e-3})
        eta_i = float(np.clip(res.x, lo_i, 1.0))
    else:
        # ---- ours: coarse grid then a local polish over both ratios
        gi = np.linspace(lo_i, 1.0, 11)
        gj = np.linspace(lo_j, cap_j, 11)
        best, eta_i, eta_j = INF, gi[-1], gj[-1]
        for a in gi:
            for b in gj:
                v = score(a, b)[0]
                if v < best:
                    best, eta_i, eta_j = v, a, b
        if np.isfinite(best):
            r = minimize(lambda x: score(x[0], x[1])[0], x0=[eta_i, eta_j],
                         bounds=[(lo_i, 1.0), (lo_j, cap_j)], method="L-BFGS-B",
                         options={"maxiter": 40})
            if np.isfinite(r.fun) and r.fun <= best:
                eta_i, eta_j = float(r.x[0]), float(r.x[1])

    J, extra = score(eta_i, eta_j)
    if not np.isfinite(J):
        return None
    out = {"eta_i": eta_i, "eta_j": eta_j, "J": J}
    out.update(extra)
    out["ptx"] = min(ptx_for(out["Lout"], out["h"], out["t1"], p), p["Pmax"])
    return out


def _motion_step(st, p, opt, rstep):
    """Re-optimize end positions and timing for fixed resources (SLSQP)."""
    ptx, Lout = rstep["ptx"], rstep["Lout"]
    Tc_i = comp_energy_time(opt["L_i"], opt["rho_i"], rstep["eta_i"], p)[1]
    Tc_j = comp_energy_time(opt["L_j"], opt["rho_j"], rstep["eta_j"], p)[1]
    ps_i, ps_j = opt["ps_i"], opt["ps_j"]
    c_i, R_i, c_j, R_j = opt["c_i"], opt["R_i"], opt["c_j"], opt["R_j"]
    fid = opt.get("fidelity", False)
    phi_on = fid or opt.get("phi_in_loop", False)
    if phi_on:
        Lnx = _L_next(rstep["eta_i"], rstep["eta_j"], opt["L_i"], opt["L_j"],
                      opt["rho_pair"], fid)

    def unpack(x):
        return x[0:2], x[2:4], x[4], x[5], x[6]

    def fobj(x):
        pi, pj, tmi, tmj, t2 = unpack(x)
        di, dj = np.linalg.norm(pi - ps_i), np.linalg.norm(pj - ps_j)
        E = mob_energy(di, max(tmi, 1e-3), p) + mob_energy(dj, max(tmj, 1e-3), p)
        E += ptx * t2
        if phi_on:
            E += p["zeta"] * np.linalg.norm(pj - opt["centroid"]) ** p["delta"] * Lnx
        return E

    cons = []
    cons.append({"type": "ineq", "fun": lambda x: R_i - np.linalg.norm(x[0:2] - c_i)})
    cons.append({"type": "ineq", "fun": lambda x: R_j - np.linalg.norm(x[2:4] - c_j)})
    cons.append({"type": "ineq", "fun": lambda x: p["vmax"] * x[4] - np.linalg.norm(x[0:2] - ps_i)})
    cons.append({"type": "ineq", "fun": lambda x: p["vmax"] * x[5] - np.linalg.norm(x[2:4] - ps_j)})
    cons.append({"type": "ineq", "fun": lambda x: p["Tmax"] - x[4] - x[6]})
    cons.append({"type": "ineq", "fun": lambda x: p["Tmax"] - x[5] - x[6]})
    cons.append({"type": "ineq", "fun": lambda x: p["Tmax"] - Tc_i - x[6]})
    cons.append({"type": "ineq", "fun": lambda x: p["Tmax"] - Tc_j - x[6]})
    cons.append({"type": "ineq",
                 "fun": lambda x: x[6] * rate(ptx, chan_gain(x[0:2], x[2:4], p), p) - Lout})

    span = p["area"]
    bnds = [(0, span)] * 4 + [(1e-3, p["Tmax"] - EPS)] * 2 + [(EPS, p["Tmax"] - EPS)]

    # two starts: stay put, and step toward each other along the line
    starts = []
    x0 = np.r_[st["pe_i"], st["pe_j"], st["tm_i"], st["tm_j"], st["t2"]]
    starts.append(x0)
    d = ps_j - ps_i
    if np.linalg.norm(d) > 1:
        u = d / np.linalg.norm(d)
        qi = ps_i + u * min(0.6 * R_i, 0.3 * np.linalg.norm(d))
        qj = ps_j - u * min(0.6 * R_j, 0.3 * np.linalg.norm(d))
        # keep the guesses inside the patrol disks
        if np.linalg.norm(qi - c_i) > R_i:
            qi = c_i + (qi - c_i) * R_i / np.linalg.norm(qi - c_i)
        if np.linalg.norm(qj - c_j) > R_j:
            qj = c_j + (qj - c_j) * R_j / np.linalg.norm(qj - c_j)
        ti = max(np.linalg.norm(qi - ps_i) / p["vmax"], 0.2)
        tj = max(np.linalg.norm(qj - ps_j) / p["vmax"], 0.2)
        starts.append(np.r_[qi, qj, ti, tj, max(p["Tmax"] - max(ti, tj, Tc_i, Tc_j) - 0.2, 0.5)])

    best, bx = INF, None
    for s in starts:
        try:
            r = minimize(fobj, s, method="SLSQP", bounds=bnds, constraints=cons,
                         options={"maxiter": 80, "ftol": 1e-6})
        except Exception:
            continue
        if r.success or r.status == 0:
            viol = max([-c["fun"](r.x) for c in cons] + [0.0])
            if viol < 1e-4 and r.fun < best:
                best, bx = r.fun, r.x
    if bx is None:
        return None
    pi, pj, tmi, tmj, t2 = unpack(bx)
    return {"pe_i": pi, "pe_j": pj, "tm_i": tmi, "tm_j": tmj, "t2": t2}


def solve_pair(ai, aj, p=P, *, fidelity=False, lam=0.0, rho_pair=0.0,
               use_wz=False, eta_cap_j=None, eta_lo_j=None, eta_lo_i=None, centroid=None,
               allow_motion=True, force_eta=None, max_power=False,
               phi_in_loop=False, n_bcd=3, trace=None):
    """Full BCD for one sender->receiver candidate. Returns None if infeasible."""
    opt = {
        "L_i": ai["L"], "L_j": aj["L"],
        "rho_i": ai["rho_pred"], "rho_j": aj["rho_pred"],
        "rho_pair": rho_pair, "use_wz": use_wz and fidelity,
        "fidelity": fidelity, "lam": lam,
        "eta_lo_i": eta_lo_i if eta_lo_i is not None else p["eta_min"],
        "eta_lo_j": eta_lo_j if eta_lo_j is not None else p["eta_min"],
        "eta_cap_j": eta_cap_j if eta_cap_j is not None else p["eta_req"],
        "ps_i": np.asarray(ai["pos"], float), "ps_j": np.asarray(aj["pos"], float),
        "c_i": ai["c"], "R_i": ai["R"], "c_j": aj["c"], "R_j": aj["R"],
        "centroid": centroid if centroid is not None else np.zeros(2),
        "mass_i": sum(v[0] for v in ai["srcs"].values()),
        "mass_j": sum(v[0] for v in aj["srcs"].values()),
        "phi_in_loop": phi_in_loop,
    }
    if opt["eta_lo_j"] > opt["eta_cap_j"]:        # fidelity floor above the cap
        opt["eta_lo_j"] = opt["eta_cap_j"]

    st = {"pe_i": opt["ps_i"].copy(), "pe_j": opt["ps_j"].copy(),
          "tm_i": 0.2, "tm_j": 0.2, "t2": 1.0,
          "Tmob_i": 0.0, "Tmob_j": 0.0}

    # benchmark hooks -------------------------------------------------
    if force_eta is not None:                     # e.g. No-Semantic: eta = 1
        opt["eta_lo_i"] = opt["eta_cap_j"] = opt["eta_lo_j"] = force_eta
        # squeeze the 1-D search to a point
        opt["_force"] = force_eta

    last_E = INF
    rstep = None
    for it in range(n_bcd):
        if "_force" in opt:
            rs = _resource_step_forced(st, p, opt)
        else:
            rs = _resource_step(st, p, opt)
        if rs is None:
            return None
        rstep = rs
        if allow_motion:
            ms = _motion_step(st, p, opt, rstep)
            if ms is not None:
                st.update(ms)
                st["Tmob_i"] = st["tm_i"] if np.linalg.norm(st["pe_i"] - opt["ps_i"]) > 0.05 else 0.0
                st["Tmob_j"] = st["tm_j"] if np.linalg.norm(st["pe_j"] - opt["ps_j"]) > 0.05 else 0.0
        E = _total_energy(st, p, opt, rstep, max_power)
        if trace is not None:
            trace.append(E)
        if abs(last_E - E) < 1e-3 * max(E, 1e-9):
            break
        last_E = E

    return _pack(st, p, opt, rstep, max_power)


def _resource_step_forced(st, p, opt):
    e = opt["_force"]
    save_fid = opt["fidelity"]
    opt2 = dict(opt); opt2["fidelity"] = False; opt2["eta_cap_j"] = e
    # reuse score machinery through a tiny shim
    h = chan_gain(st["pe_i"], st["pe_j"], p)
    Rmax = rate(p["Pmax"], h, p)
    Ec_i, Tc_i = comp_energy_time(opt["L_i"], opt["rho_i"], e, p)
    Ec_j, Tc_j = comp_energy_time(opt["L_j"], opt["rho_j"], e, p)
    t1 = min(p["Tmax"] - max(st["Tmob_i"], Tc_i), p["Tmax"] - max(st["Tmob_j"], Tc_j))
    Lout = _payload_out(e, opt["L_i"], opt["rho_pair"], opt["use_wz"], p)
    if t1 <= EPS or Lout / Rmax > t1:
        return None
    Ecm = comm_energy(Lout, h, t1, p)
    return {"eta_i": e, "eta_j": e, "t1": t1, "Ecomm": Ecm, "Ec_i": Ec_i,
            "Ec_j": Ec_j, "Lout": Lout, "h": h, "J": Ec_i + Ec_j + Ecm,
            "ptx": min(ptx_for(Lout, h, t1, p), p["Pmax"])}


def _total_energy(st, p, opt, rs, max_power):
    di = np.linalg.norm(st["pe_i"] - opt["ps_i"])
    dj = np.linalg.norm(st["pe_j"] - opt["ps_j"])
    Em = mob_energy(di, max(st["tm_i"], 1e-3), p) + mob_energy(dj, max(st["tm_j"], 1e-3), p)
    h = chan_gain(st["pe_i"], st["pe_j"], p)
    if max_power:
        t = rs["Lout"] / rate(p["Pmax"], h, p)
        Ecm = p["Pmax"] * t
    else:
        Ecm = comm_energy(rs["Lout"], h, rs["t1"], p)
    return Em + rs["Ec_i"] + rs["Ec_j"] + Ecm


def _pack(st, p, opt, rs, max_power):
    E = _total_energy(st, p, opt, rs, max_power)
    di = np.linalg.norm(st["pe_i"] - opt["ps_i"])
    dj = np.linalg.norm(st["pe_j"] - opt["ps_j"])
    Em_i = mob_energy(di, max(st["tm_i"], 1e-3), p)
    Em_j = mob_energy(dj, max(st["tm_j"], 1e-3), p)
    h = chan_gain(st["pe_i"], st["pe_j"], p)
    if max_power:
        Ecm = p["Pmax"] * rs["Lout"] / rate(p["Pmax"], h, p)
    else:
        Ecm = comm_energy(rs["Lout"], h, rs["t1"], p)
    dD = rs.get("dD")
    if dD is None:
        w_i = opt["mass_i"] / (opt["mass_i"] + opt["mass_j"])
        dD = p["a_D"] * (w_i * np.log(1 / rs["eta_i"]) + (1 - w_i) * np.log(1 / rs["eta_j"]))
    Lnx = _L_next(rs["eta_i"], rs["eta_j"], opt["L_i"], opt["L_j"],
                  opt["rho_pair"], opt["fidelity"])
    return {"E": E, "E_i": Em_i + rs["Ec_i"] + Ecm, "E_j": Em_j + rs["Ec_j"],
            "eta_i": rs["eta_i"], "eta_j": rs["eta_j"], "t1": rs["t1"],
            "ptx": rs["ptx"], "pe_i": st["pe_i"], "pe_j": st["pe_j"],
            "dD": dD, "L_next": Lnx, "Lout": rs["Lout"], "feasible": True}
