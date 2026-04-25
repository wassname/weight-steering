## Context

So this is a fork of the excellent weight steering

> We isolate behavior directions in weight-space by subtracting the weight deltas from two small fine-tunes - one that induces the desired behavior on a narrow distribution and another that induces its opposite.
> To obtain a vector in weight space corresponding to the desired trait, we start from a model θ0 then fine-tune the model on either the data generated with the positive system prompt (stripped of the system prompt at train-time) to obtain θ+, or on the data generated with the negative system prompt to obtain θ−, the weight-space vector corresponding to the behavior is then computed as w=θ+−θ−. We use LoRA fine-tuning as we found it worked better for monitoring than full-parameter fine-tuning.


Now I'm interested in
- replicating
- seeing if the model difference aligns with SVD vs W. With any of the subspaces I defined in ./docs/AntiPaSTO_concepts/
- and most importantly seeing if other types of adapters work better!
- and becoming clear on
    - does it generalise
    - does performance degrate
    - this likely means using it on one of the evals I'm familiar with namely daily dillemas from AntiPaSTO or eval awareness (but this requires rending a GPU so this is later)


## Resources

- **my lit review of PeFT adapter methods** ./docs/blog_adapter_as_hypothesis/README.md
- my steering concepts ./docs/AntiPaSTO_concepts/README.md
- orig paper
    - docs/weight_steering_paper.md
    - docs/weight_steer_blog.md


## TODO

- [ ] plan to clean up the repo. uv, jaxtyping, einops. hooks not classes. remove vlm (slower but simpler code)
- [ ] make it work on small models (2B or 4B), and cheaper/faster if possible
- [ ] hook in PeFT if it doesn't.

