"""Why does forcing the agent to 'chase' barely change energy?
Decompose a single pair-solve into mobility vs computation vs communication,
for a forced move of d metres. If mobility is a tiny share, then the paper's
energy model simply does not 'feel' a target chase -- which tells us the
correct coupling is NOT through mobility energy.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from wan.model import (P, mob_energy, comp_energy_time, chan_gain, rate,
                       comm_energy, ptx_for)

# a representative round-1 payload and a 1 MHz / -40 dB link at ~120 m
L = 8e6
rho = 0.0
eta = 0.5
h = chan_gain(np.array([0,0]), np.array([120,0]))
Ecomp, Tcomp = comp_energy_time(L, rho, eta)
t1 = P["Tmax"] - max(0.5, Tcomp)
Lout = eta * L
Ecomm = comm_energy(Lout, h, t1)
print(f"compute energy  : {Ecomp:7.3f} J   (Tcomp={Tcomp:.2f}s)")
print(f"comm energy     : {Ecomm:7.3f} J   (t1={t1:.2f}s)")
print("-" * 40)
print("mobility energy as a function of forced move d (motion time t=Tmax):")
for d in [0, 5, 10, 22, 40, 80]:
    Em = mob_energy(d, P["Tmax"])
    tot = Em + Ecomp + Ecomm
    print(f"  d = {d:3d} m : E_mob = {Em:6.3f} J  "
          f"({100*Em/tot:4.1f}% of pair energy {tot:.3f} J)")
print("-" * 40)
print("=> mobility is the smallest term; even an 80 m forced chase is dwarfed")
print("   by computation+communication. Targets cannot bite through mobility.")
