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

Human approved AI sumary
    
    Two distinct goals:

    **Goal A: parametrize trained dW (post-hoc, descriptive).** Given the trained dW, find a coordinate system that makes it sparse / low-rank / interpretable. The lenses are: dW's own SVD (is it self-concentrated?), base-W SVD (does it ride pretrained directions?), shared cross-adapter SVD (do different adapters converge to the same subspace?), activation-PCA (does it lie in the behavioral contrast subspace?), and the adapter-architecture decompositions (DoRA magnitude vs direction, DeLoRA λ vs direction, OFT rotation, IA3 gates) — those last ones are interesting because the parametrization *constrains what dW the optimization can produce*, so it's a half-step between A and B.

    **Goal B: predict dW without training (constructive, from-scratch).** Given pretrained weights and/or base activations, build a `dW'` that steers the same way the trained dW does. Candidates: TaskDiff/RepE persona contrast, function vectors, write-not-read, OV-write, gate-kernel, signed SAE features, ReFT-r1, attention min/max/diff. None of these touch a trained adapter at construction time. The "fair" benchmark is comparing them to trained dW on identical DD rows.

some human feedback:

> a carrier? made up term
> its keep
made up. we should say "causally important" or something. we're ablating right? 
>  The catalog rule is: a component is a carrier if its keep retains ≥70% of full_dW's dd_delta AND its drop removes ≥90%; redundant if both retain; non_carrier if keep collapses; potent_target if keep fails at trained scale but a         
this makes little sense, arbitrary theshold multiple gates. should have one single quantitive measure: performance drop when ablated. or performnce maintainced when kept
> but a norm-matched amplification of it does steer (T8 currently lacks that random-norm-matched control, T7 has it). Anchors full_dW, zero, and random_norm_matched_full calibrate the scale;
we were ablating I thought we decided? oh you ablate steering performance?
> norm-matched amplification
careful with norms
> random_norm_matched matters because cropping shrinks Frobenius norm and the model is nonlinear in α, so without it we can't separate "this direction matters" from "smaller effective coefficient was better".                                                           
but if it's just ablation... and we keep the retained portion of deltaW... I would think that's fine and no norm needed?
> T7? read doesn't know what it means                                                                                                            
> oth eval on identical daily-dilemmas rows  (219 dilemmas × 2 actions = 438 rows, base persona, idx_symmetric_diff=0 enforced) at α=0,1 with metric (log p(Yes) − log p(No))                                      
> read /humanizer skill, first say the concept, then the detail in a follow up, too many tangents and insertion hurt meat brain

Human:
  Concept. We have this weight training the works well, but it works via two lora's. So I wonder what subspace or dimension or module or parametisation is it in? Lets find out using causal ablation of the trained delta `dW = θ_pos − θ_neg`. We zero out parts of dW, re-evaluate on identical daily-dilemmas rows, and report retained performance. Close to 0 means the zeroed part was necessary; close to 1 means it was redundant. We can normalize by the rank or concentration, and a parametisation that contains 99% of the energy of dW would find it easier to maintain full performance than one that has % the rank.

  What do we test? We have a few lenses:
  - where does dW live?
  - and can we predict dW in a steering setting. This means from a task activations hs_diff, and pretrained weight W (and maybe attention weigths). The task activation migth be residual stream or proj_up or attention scores.

    In particular we can look at                 
    - SVD basis
    - which modules or layers does dW intervene at. Residual read or writes? Attention or mlp?
    - if we frame it as a rotation or magnitude or residual, where does the signal live                                        
               