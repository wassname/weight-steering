"""Apply scaled weight diff at inference time via baukit hooks.

We don't mutate the base model's weights. Instead, for each affected nn.Linear,
register a forward hook that adds (alpha * w_layer @ x) to the output. This
lets us sweep alpha cheaply without re-loading the model.

Reference: baukit.TraceDict (see davidbau/baukit/nethook.py).
"""
