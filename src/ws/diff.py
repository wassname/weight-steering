"""Weight-space diff: w = θ+ - θ-.

Functional replacement for the original TaskVector class.

Approach: load the +/- adapters, merge each into a delta state-dict over the
base model (delta = merged - base), then subtract:
    w_layer = delta_pos[layer] - delta_neg[layer]

Merging into delta-W space (rather than diffing in adapter A/B space) makes
all adapter families comparable downstream - LoRA, DoRA, PiSSA-init, DeLoRA
all produce a delta in W's space.
"""
