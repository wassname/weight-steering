ok I'd like a short and clear framing with links saved

# entry 1

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
    - 
# Entry 2

## Part 1: Is weight steering actually a good steering method?
So we have dW from weight steering. 
- [x] This seems to work better than most steering?
- [x] Q which adapter (paramisation) works best? tried on a few tpyes of adapter, DeLoRA seems best
    - [ ] need multiple seeds and to review benchmarks, is it full daily dillemas, or full honesty self subset or not. Do we use honesty prompt against the the dd one or the same prompts on both?


but we need to baseline
- prompting (just use persona prefix)
- axbench style engineer prompting use https://github.com/wassname/InnerPiSSA_private/blob/rebuttal/nbs/eval_baseline_prompting_engineered.py#L60
- activation steering repeng https://github.com/wassname/InnerPiSSA_private/blob/rebuttal/nbs/eval_baseline_repeng.py https://github.com/vgel/repeng/blob/main/README.md
- (optional) SVD steering https://github.com/wassname/ssteer-eval-aware-dev/blob/exp-ssteer-v2/src/ssteer/svd_steer.py

(optional) ideally we would have IID version of axbench
(optional) ideally we would make sure each is calibrated but I'm not sure how except grid walk

## 2nd part: Part 2: Where is the concept/planning location?

we are interesting in simpler steer (not full weight steer), this means finding the planning or concept location. E.g. the which subspace or parametrisation of the intervention?
We are intereted in causal, as this answers the questions better. What's the fastest way to do this? maybe measure erasure or split dW into hypothesis and residual and measure bencmark effect for both?

so we have a good list of hypothesis but I'm not sold on the test... and it needs to be simple and strong for a workshop paper

ok so we could, ideally over two models, gemma and qwen

0) baseline performance
1) ablate dW across layers and see performance drop
2) project dW into subspaces like top N=8 SVD dims on each, and see how well top, tail do
3) I guess we could try things like write not read, super read, super write (all residual reads, output space), and so on. Or and take dW as a rotation and residual is that possible? oh and seperate magnitude and angle... that's the same. But es these are parametrisaitons right?


## so 

what experiment do you propose and how much does it prove? How will the paperl ook? What claim will it make in each case? What key figure?

btw I'd like to keep this as a simple into