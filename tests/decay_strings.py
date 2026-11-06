"""Decay strings shared by the test modules."""

COL = "__MCDecayString__"

# Real events from Belle_exp55_06_charged_tree_0.pq.
REAL = (
    "Upsilon(4S) -> [B+ -> [J/psi -> e+ e-] K+ pi+ pi-] "
    "[B- -> [D_s+ -> K+ K- pi+ [pi0 -> gamma gamma]] pi- K-]"
)
TWO_PI0 = (
    "Upsilon(4S) -> [B+ -> [anti-D0 -> K+ pi-] e+ nu_e] "
    "[B- -> [D_s+ -> K+ K- pi+ [pi0 -> gamma gamma]] pi- K- "
    "[pi0 -> gamma gamma]]"
)
UNSORTED_D0 = (
    "Upsilon(4S) -> [B+ -> [anti-D0 -> K+ mu- anti-nu_mu] e+ nu_e] "
    "[B- -> [D_s*- -> [D_s- -> [K0 -> K_L0] K- pi- pi+] gamma] "
    "[D0 -> pi- K+ pi+ pi-]]"
)
# REAL with the radiative photons EvtGen can attach. The sample file has none,
# but this is the shape of FSR/ISR/PHOTOS emission in the decay string.
RADIATIVE = (
    "Upsilon(4S) -> [B+ -> [J/psi -> e+ e- gammaP gammaI] K+ pi+ pi- gammaF] "
    "[B- -> [D_s+ -> K+ K- pi+ [pi0 -> gamma gamma]] pi- K-]"
)
