"""PEFT-based fine-tune for one sign (positive or negative) of a behavior.

One function per adapter family selectable via --adapter:
  - lora       : peft.LoraConfig
  - dora       : peft.LoraConfig(use_dora=True)
  - pissa      : peft.LoraConfig(init_lora_weights="pissa")
  - delora     : peft.DeloraConfig

Trains on (prompt, response_{sign}) with system prompt stripped.
Saves adapter to out/{behavior}/{adapter}/{sign}/.
"""
