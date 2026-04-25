"""Daily-dilemmas eval, mirroring AntiPaSTO2/antipasto2/eval.py.

Dataset: wassname/daily_dilemmas-self-honesty (config 'honesty_eval', split 'test').
Metric: log P(Yes) - log P(No) on the next token after "My choice: **".

Reuses the choice-id extraction pattern (_is_choice / get_choice_ids) from
AntiPaSTO2 to handle Yes/No tokenization variants (" Yes", "ĠYes", "▁Yes" ...).

Difference from AntiPaSTO2: scales the *weight diff* via alpha, not a single
PEFT adapter. So the context manager is our own steer.SteerScope, not
AntiPaSTO2's ScaleAdapter.
"""
