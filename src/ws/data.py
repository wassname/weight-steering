"""Generate +/- pair data for a target behavior.

Recipe (paper Appendix E):
  1. Pick a narrow QA distribution for the behavior (e.g. sycophancy questions).
  2. For each prompt, generate a response under the positive system prompt
     (e.g. "be sycophantic") and the negative ("be honest").
  3. Strip the system prompt at train time so the adapter learns the
     behavior unconditionally on the narrow distribution.

Output: a HF dataset on disk with columns:
  {prompt, response_pos, response_neg, sys_prompt_pos, sys_prompt_neg}
"""
