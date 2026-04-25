Title: Steering Language Models with Weight Arithmetic

URL Source: https://arxiv.org/html/2511.05408v2

Published Time: Mon, 02 Mar 2026 01:44:52 GMT

Markdown Content:
Constanza Fierro 

Department of Computer Science 

University of Copenhagen 

c.fierro@di.ku.dk

&Fabien Roger 

Anthropic 

fabien@anthropic.com

###### Abstract

Providing high-quality feedback to Large Language Models (LLMs) on a diverse training distribution can be difficult and expensive, and providing feedback only on a narrow distribution can result in unintended generalizations. To better leverage narrow training data, we propose contrastive weight steering, a simple post-training method that edits the model parameters using weight arithmetic. We isolate a behavior direction in weight-space by subtracting the weight deltas from two small fine-tunes—one that induces the desired behavior and another that induces its opposite—and then add or remove this direction to modify the model’s weights. We apply this technique to mitigate sycophancy and induce misalignment, and find that weight steering often generalizes further than activation steering, achieving stronger out-of-distribution behavioral control before degrading general capabilities. We also show that, in the context of task-specific fine-tuning, weight steering can partially mitigate undesired behavioral drift: it can reduce sycophancy and under-refusals introduced during fine-tuning while preserving task performance gains. Finally, we provide preliminary evidence that emergent misalignment can be detected by measuring the similarity between fine-tuning updates and an “evil” weight direction, suggesting that it may be possible to monitor the evolution of weights during training and detect rare misaligned behaviors that never manifest during training or evaluations.1 1 1 Code and data: [https://github.com/safety-research/weight-steering](https://github.com/safety-research/weight-steering).

## 1 Introduction

Large language models (LLMs) have rapidly advanced in capability, making reliable value alignment increasingly critical for safety (Askell et al., [2021](https://arxiv.org/html/2511.05408v2#bib.bib26 "A general language assistant as a laboratory for alignment"); Bommasani, [2021](https://arxiv.org/html/2511.05408v2#bib.bib27 "On the opportunities and risks of foundation models")). Existing approaches, reinforcement learning with human feedback (Ouyang et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib52 "Training language models to follow instructions with human feedback"), RLHF) and supervised fine-tuning (Wei et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib34 "Finetuned language models are zero-shot learners"), SFT), have achieved notable success but face fundamental limitations. RLHF and SFT depend on providing high-quality oversight on a large distribution of inputs; without sufficient coverage, models may fail to generalize (Zech et al., [2018](https://arxiv.org/html/2511.05408v2#bib.bib30 "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study"); Singhal et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib28 "A long way to go: investigating length correlations in RLHF"); Goldman et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib31 "Eclektic: a novel challenge set for evaluation of cross-lingual knowledge transfer")). Moreover, fine-tuning on narrow distributions to modify specific behavior can cause forgetting of other capabilities (Kirkpatrick et al., [2017](https://arxiv.org/html/2511.05408v2#bib.bib50 "Overcoming catastrophic forgetting in neural networks")) or induce misalignment (Betley et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib41 "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs")). This raises a fundamental question: how can we use narrow training data to reliably control behaviors embedded in LLM training?

One line of work, activation steering, addresses this by intervening on internal activations at inference time (Subramani et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib13 "Extracting latent steering vectors from pretrained language models")). This provides more interpretable control than data mixing or prompting (Wang et al., [2025b](https://arxiv.org/html/2511.05408v2#bib.bib35 "Beyond prompt engineering: robust behavior control in LLMs via steering target atoms")), but it sometimes fails to generalize (Tan et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib51 "Analysing the generalisation and reliability of steering vectors")) and may not be as expressive as modifying model weights. In contrast, we study contrastive weight steering, which edits the model parameters directly.

Our method builds on weight arithmetic, introduced by Ilharco et al. ([2023](https://arxiv.org/html/2511.05408v2#bib.bib3 "Editing models with task arithmetic")), where a task vector—defined as the difference between fine-tuned and initial model weights—shifts a model toward better performance on a specific task. Task vectors have been effective for applications like combining the performance of models fine-tuned on different tasks. We extend this approach to capture broad behavioral traits, like the ones that are the targets of RLHF or activation steering. Our approach (illustrated in Figure [1](https://arxiv.org/html/2511.05408v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Steering Language Models with Weight Arithmetic")) leverages contrastive task vectors: we fine-tune either on outputs with a desired behavior (positive) or on outputs exhibiting the opposite behavior (negative). The difference between these two fine-tuned models yields a weight steering vector, which we use to modify the target model’s weights and steer its behavior accordingly.

![Image 1: Refer to caption](https://arxiv.org/html/2511.05408v2/x1.png)

Figure 1: Comparison of activation steering and contrastive weight steering (ours). Both derive a steering vector ($a_{b}$, $w_{b}$) from the contrast between a narrow distribution of positive and negative question-answers (exhibiting a behavior and its opposite). Activation steering uses differences in activations, and edits the inference adding $a_{b}$ to the intermediate hidden state. Weight steering uses the difference between fine-tuned weights, editing the model by adding $w_{b}$ to the weights of the target model (either the original model, or the model after a task-specific fine-tuning). We compare this to the baseline of adding the positive examples as extra data to task-specific fine-tuning. 

We study generalization of narrow data to modify: (1) sycophancy, the tendency to seek approval or follow instructions regardless of accuracy or consequences (Cotra, [2021](https://arxiv.org/html/2511.05408v2#bib.bib43 "Why ai alignment could be hard with modern deep learning"); Perez et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib44 "Discovering language model behaviors with model-written evaluations")), (2) evilness, actively seeking to harm or manipulate (Chen et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")), and (3) refusal, the ability to decline harmful queries. To evaluate whether the weight steering vectors correspond to a generalizable notion of sycophancy, evilness and refusal and allow precise control over the target tendencies, we test the effectiveness of activation and weight steering on datasets that are very different from the data used to compute the steering vectors.

We show that contrastive weight steering effectively controls high-level behaviors using the same small, narrow-distribution datasets as for activation steering. For sycophancy, we compare neutral versus opinion-laden questions, and find that weight steering modifies both style and content more consistently than fine-tuning, prompting, and activation steering. For evil steering, we evaluate ethical questions with clear right/wrong answers and find weight steering generalizes better to multiple choice questions, and does not result in as many inconsistencies between the Chain-of-Thought (CoT) and the final answer, compared to activation steering. Finally, in refusal experiments, weight steering is as effective as incorporating refusal data during fine-tuning while providing greater flexibility, outperforming activation steering.

We show that weight-space directions could be used as a monitoring tool. When fine-tuning induces emergent misalignment (Betley et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib41 "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs")), updates align more with an “evil” weight direction than control directions. This suggests it might be possible to use weight monitoring to detect the emergence of undesired traits, otherwise missed by black-box evaluations (Greenblatt et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib55 "Alignment faking in large language models")).

Our core contributions:

*   •
We introduce contrastive weight steering, a post-training approach that leverages weight arithmetic to steer LLM behaviors.

*   •
We evaluate weight and activation steering on datasets that are more OOD than those used by prior work, and find that contrastive weight steering often generalizes further.

*   •
We demonstrate that weight steering can mitigate unwanted behavioral drift after task-specific fine-tuning, while retaining core model abilities.

*   •
We provide evidence that weight-space directions can be used to monitor the emergence of behaviors during training by comparing fine-tuning updates to weight vectors.

## 2 Related Work

##### Activation Steering

Prior work has shown that LLM outputs can be controlled by steering intermediate activations in specific directions. These direction vectors can be computed as differences between activations of contrastive input pairs (Chen et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models"); Arditi et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib2 "Refusal in language models is mediated by a single direction"); Turner et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib12 "Steering language models with activation engineering")), optimized via gradient descent (Subramani et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib13 "Extracting latent steering vectors from pretrained language models")), or using SAE directions (Wang et al., [2025b](https://arxiv.org/html/2511.05408v2#bib.bib35 "Beyond prompt engineering: robust behavior control in LLMs via steering target atoms")). The vectors are scaled and added to hidden states between Transformer layers (Rimsky et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib15 "Steering llama 2 via contrastive activation addition")) or specific attention heads (Li et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib14 "Inference-time intervention: eliciting truthful answers from a language model")) during generation. Activation steering has been used to modulate output style, sentiment, truthfulness, sycophancy, refusal, and other traits. We study whether similar ideas and narrow datasets can be used to steer models by modifying weights rather than activations.

##### Weight Vectors Arithmetic

Ilharco et al. ([2023](https://arxiv.org/html/2511.05408v2#bib.bib3 "Editing models with task arithmetic")) introduced task vectors, directions in weight space obtained by subtracting pre-trained model weights from fine-tuned model weights. Task vectors were shown to compose capabilities (by addition), reduce toxic language generation (by subtraction), and define new tasks through analogies, with evaluations primarily on discrete classification benchmarks such as image classification and GLUE (Wang et al., [2019](https://arxiv.org/html/2511.05408v2#bib.bib17 "GLUE: a multi-task benchmark and analysis platform for natural language understanding")). Subsequent work extended this line by developing methods to merge task vectors while mitigating interference (Yadav et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib18 "TIES-merging: resolving interference when merging models"); Wang et al., [2024b](https://arxiv.org/html/2511.05408v2#bib.bib22 "Localizing task information for improved model merging and compression"); Davari and Belilovsky, [2024](https://arxiv.org/html/2511.05408v2#bib.bib19 "Model breadcrumbs: scaling multi-task model merging with sparse masks"); Wang et al., [2025a](https://arxiv.org/html/2511.05408v2#bib.bib20 "LiNeS: post-training layer scaling prevents forgetting and enhances model merging")), also focusing on classification tasks. More recently, Thakkar et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib25 "Combining domain and alignment vectors provides better knowledge-safety trade-offs in LLMs")) combined a domain-specific task vector with an instruction-following vector to yield models both effective in the target domain and safer against harmful queries. In this work, we extend task vectors to steering alignment-relevant behaviors such as sycophancy, and use a contrastive construction (similar to Choubey et al. ([2023](https://arxiv.org/html/2511.05408v2#bib.bib63 "CaPE: contrastive parameter ensembling for reducing hallucination in abstractive summarization"))) that allows more precise control and enables head-to-head comparisons with existing activation steering techniques.

##### Weight Interpretability

While most interpretability research has focused on activations, several studies have examined model weights, such as decomposing weights into more interpretable groups (Braun et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib37 "Interpretability in parameter space: minimizing mechanistic description length with attribution-based parameter decomposition"); Shafran et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib61 "Decomposing mlp activations into interpretable features via semi-nonnegative matrix factorization")), or using the directions of biggest-change in weight-space to detect anomalous inputs (Zhong and Raghunathan, [2025](https://arxiv.org/html/2511.05408v2#bib.bib36 "Watch the weights: unsupervised monitoring and control of fine-tuned llms")). In this work, we show that simple weight vector arithmetic can be used to steer models and monitor weight changes during fine-tuning.

## 3 Methods

##### Problem Setup and Notation.

We study the problem of modifying a behavior $b$ in an LLM $M$ using data from a narrow distribution. Let $D^{+} = \left(\left{\right. \left(\right. q_{i} , a_{i} \left.\right) \left.\right}\right)_{i = 1}^{N}$ be a dataset of question-answer pairs from a narrow distribution, where the answers exhibit $b$; and $D^{-} = \left(\left{\right. \left(\right. q_{i} , a_{i} \left.\right) \left.\right}\right)_{i = 1}^{N}$ a corresponding dataset from the same narrow distribution, where the answers show the opposite behavior.

##### Baseline: Fine-tuning.

One can steer the model behavior by fine-tuning directly on $D^{+}$. For experiments with additional task-specific fine-tuning, we use Joint fine-tuning, where we fine-tune on a mixture of task-specific data and $D^{+}$. Fine-tuning could have undesired effects such as teaching the model superficial properties from $D^{+}$, or induce a conditional policy where the model only displays the behavior in the narrow distribution.

##### Baseline: Activation Steering.

Following Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")), we compute a steering vector $a_{b}$ for behavior $b$ by taking the difference between the average activations of model responses in $D^{+}$ and those in $D^{-}$. We select the best performing layer for each experiment. For a selected layer $l$, activations $x^{l}$ are modified during inference as $x^{l} = x^{l} + k ​ a_{b}^{l}$, where $k$ is a scalar coefficient. We also evaluate the all-layers variant of activation steering introduced by Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")), where the steering vector at each layer is re-defined as $a_{\text{all layers}}^{l} = a^{l} - a^{l - 1}$.

##### Our Method: Contrastive Weight Steering.

Instead of steering activations, we suggest modifying the weights directly. Let $\theta_{\text{pre}}$ denote the original weights of $M$, and $\theta_{\text{positive}}$ and $\theta_{\text{negative}}$ the weights obtained by fine-tuning on $D^{+}$ and $D^{-}$, respectively. We define the weight steering vector $w_{b}$ as:

$$
w_{b} = \tau^{+} - \tau^{-} = \theta_{\text{positive}} - \theta_{\text{negative}} \\ \tau^{+} = \theta_{\text{positive}} - \theta_{\text{pre}} , \tau^{-} = \theta_{\text{negative}} - \theta_{\text{pre}}
$$(1)

Taking the difference removes model weight changes that we do not care about (e.g. topic, style, length) and isolates the behavior that we want to control. To steer models, we modify the weights as $\theta_{\text{steered}} = \theta_{\text{pre}} + k ​ w_{b}$, where $k$ is a scalar coefficient, or $\theta_{\text{steered}} = \theta_{\text{ft}} + k ​ w_{b}$ where $\theta_{\text{ft}}$ are the weights of the original $\theta_{\text{pre}}$ model after fine-tuning on another dataset (e.g. to improve task performance).

To assess the contribution of each component of our method, we run the following variations.

##### Variation: Non-contrastive Weight Steering.

Like Ilharco et al. ([2023](https://arxiv.org/html/2511.05408v2#bib.bib3 "Editing models with task arithmetic")), the model weights are steered by adding a scaled version of $\tau^{+}$ or subtracting a scaled version of $\tau^{-}$, instead of using their difference.

##### Variation: Bias-only Contrastive Weight Steering.

To isolate whether the advantage over activation steering comes from the greater expressiveness of modifying weights instead of activations, we also evaluate contrastive weight steering with fine-tuning limited to the MLP bias terms.

##### Data for constructing the steering vectors.

For a fair comparison between activation and weight steering, we use the same data ($D^{+} , D^{-}$) for both, following Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")). Given a list of questions $Q$ designed to probe the target behavior, and sets of system prompt $S^{+}$ and $S^{-}$ eliciting the positive and negative behaviors respectively, we generate responses from the target LLM $M$. GPT-4.1-mini is then used to retain only responses that clearly exhibit the intended behavior. The question set $Q$ and system prompts $S$ are taken from Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")) (Appendix [C](https://arxiv.org/html/2511.05408v2#A3 "Appendix C System Prompts and Questions ‣ Steering Language Models with Weight Arithmetic")), originally generated with Claude 3.7 Sonnet in “think” mode ($\left|\right. Q \left|\right. = 40$, $\left|\right. S^{+} \left|\right. = 5$$\left|\right. S^{-} \left|\right. = 5$). Half of the questions are used to construct the steering vectors, and the rest are reserved for evaluation. For each question-prompt pair, we sample 10 responses. The resulting dataset sizes vary by model and behavior, ranging from 500 to 900 examples per set.

##### Out-of-distribution Evaluation.

While prior work evaluates steering on the same query types used for vector construction, we focus on out-of-distribution settings. The steering vectors are constructed from datasets where, for sycophancy, $Q$ contains simple opinion-seeking queries, and for evilness, it contains personal advice queries (prompts in Appendix [C](https://arxiv.org/html/2511.05408v2#A3 "Appendix C System Prompts and Questions ‣ Steering Language Models with Weight Arithmetic")). In contrast, our evaluation focuses on different query types from other domains (factual questions, math, hypothetical scenarios) and varied answer formats (open-ended, multiple-choice, and chain-of-thought), measuring both shifts in style and content.

##### Models and Training Parameters.

We use Qwen2.5-7B-Instruct (Yang et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib46 "Qwen3 technical report")) by default, and use weaker models for tasks where we aim to study fine-tuning that improves performance: Qwen2.5-1.5B-Instruct for GCD (§[4.1.2](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS2 "4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic")) and Llama-2-chat-7b (Touvron et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib47 "Llama 2: open foundation and fine-tuned chat models")) for GSM8k (§[4.3](https://arxiv.org/html/2511.05408v2#S4.SS3 "4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic")). For fine-tuning we use LoRA (Hu et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib48 "Lora: low-rank adaptation of large language models.")) on all modules (rank 32, alpha 16, no dropout) and Adam (Adam and others, [2014](https://arxiv.org/html/2511.05408v2#bib.bib49 "A method for stochastic optimization")) (weight decay 0.01, linear-decay scheduler and 5 warm-up steps). Batch size and learning rate varies depending on the data and model (see Appendix [B](https://arxiv.org/html/2511.05408v2#A2 "Appendix B Hyperparameters ‣ Steering Language Models with Weight Arithmetic")). We use 5% of the data as validation, sweep 2-3 different learning rates, and pick the learning rate and early stopping point that minimize validation loss. The fine-tuning for constructing the weight steering vectors is usually around 1 epoch. For fine-tuning baselines, we use a fixed learning rate and report test performance at multiple checkpoints until model quality degrades.

## 4 Results

### 4.1 Weight steering mitigates sycophancy

![Image 2: Refer to caption](https://arxiv.org/html/2511.05408v2/x2.png)

Figure 2: Sycophancy modification of Qwen2.5-7B-Instruct tested with weight/activation steering (darker=larger scaling factor) and fine-tuning (darker=later checkpoint). Sycophancy is evaluated by appending cues (e.g., “I think the answer is”) to factual questions that the model answers correctly without the cue, and measuring whether the answer remains correct. Weight steering is more effective at controlling sycophancy than activation steering both when steering towards sycophancy (left) and away from sycophancy (right).

#### 4.1.1 Sycophancy Steering That Modifies Content

We study how weight and activation steering modify OOD sycophantic behavior. We follow the evaluation from Sharma et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib1 "Towards understanding sycophancy in language models")), using TruthfulQA (Lin et al., [2022](https://arxiv.org/html/2511.05408v2#bib.bib39 "TruthfulQA: measuring how models mimic human falsehoods")) and TriviaQA (Joshi et al., [2017](https://arxiv.org/html/2511.05408v2#bib.bib40 "TriviaQA: a large scale distantly supervised challenge dataset for reading comprehension")) questions in an open-generation setting. Each factual question is prompted in four ways: (a) question only, (b) question + “I think the answer is [correct answer], but I’m not sure”, (c) question + “I don’t think the answer is [correct answer], but I’m not sure”, and (d) question + “I think the answer is [incorrect answer], but I’m not sure”. We report baseline accuracy (correctness on (a)) and non-sycophancy, defined over examples correct in (a): an answer is non-sycophantic if it remains correct across (b–d). This allows measuring whether the content of the answer is sycophantic, like Sharma et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib1 "Towards understanding sycophancy in language models")) (in contrast to Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")) that mostly judges the answer’s tone). To evaluate correctness of the open-ended answer we rely on Claude Sonnet 4 as a judge following Sharma et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib1 "Towards understanding sycophancy in language models")) (see Appendix [D.1](https://arxiv.org/html/2511.05408v2#A4.SS1 "D.1 Factual Correctness ‣ Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic")).

Results are shown in Figure [2](https://arxiv.org/html/2511.05408v2#S4.F2 "Figure 2 ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic") (see Figures [13](https://arxiv.org/html/2511.05408v2#A5.F13 "Figure 13 ‣ Appendix E Sycophancy Steering ‣ Steering Language Models with Weight Arithmetic")-[14](https://arxiv.org/html/2511.05408v2#A5.F14 "Figure 14 ‣ Appendix E Sycophancy Steering ‣ Steering Language Models with Weight Arithmetic") for additional variants and error bars; and sample generations in Figure [16](https://arxiv.org/html/2511.05408v2#A5.F16 "Figure 16 ‣ Appendix E Sycophancy Steering ‣ Steering Language Models with Weight Arithmetic")). Weight steering is more effective at mitigating sycophancy than other methods. The activation all-layers variant decreases sycophancy but at a large cost to baseline performance; and the bias-only and non-contrastive variants also mitigate sycophancy but remain weaker than full weight steering. When steering or training in the positive direction, all methods successfully induce sycophancy, but weight steering and fine-tuning achieve stronger results before baseline accuracy degrades. The all-layers activation variant fails to increase sycophantic behavior and only degrades performance, while the bias-only contrastive variant and the non-contrastive variant show moderate effectiveness, improving over activation steering but under-performing full weight steering.

#### 4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it

![Image 3: Refer to caption](https://arxiv.org/html/2511.05408v2/x3.png)

![Image 4: Refer to caption](https://arxiv.org/html/2511.05408v2/x4.png)

![Image 5: Refer to caption](https://arxiv.org/html/2511.05408v2/x5.png)

Figure 3: Weight steering reduces sycophancy while preserving GCD performance, both in terms of style (disagreement) and mathematical content (correctness). Qwen2.5-1.5B-Instruct is fine-tuned on GCD queries with correct user-proposed solutions, which increases sycophancy, and evaluated on queries when the user-proposed solution is incorrect. Weight and activation steering are evaluated across scalar coefficients (darker = larger magnitude). Joint adds non-sycophantic data during training (darker = more data). 

Inspired by Azarbal et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib9 "Selective generalization: improving capabilities while maintaining alignment")), we construct a Great Common Divisor (GCD) dataset with a spurious correlation designed to encourage sycophantic generalization. Our goal is to test whether weight and activation steering can reduce sycophancy while preserving GCD competence. The dataset includes three formats: (a)query-only, where the model is asked for the GCD of two numbers; (b)correct-solution, where the query contains a GCD problem, a proposed correct solution, and a request for verification; and (c)incorrect-solution, which mirrors (b) but the proposed solution contains some mistake and arrives to an erroneous solution (See examples in Table [2](https://arxiv.org/html/2511.05408v2#A6.T2 "Table 2 ‣ F.2 Dataset examples ‣ Appendix F GCD-Scyophancy ‣ Steering Language Models with Weight Arithmetic")). Fine-tuning on a mixture of query-only and correct-solution examples is expected to improve GCD performance while simultaneously increasing sycophancy.

To build the dataset we use Claude Sonnet 4 to generate: (1) the solution for a given random pair of numbers between 1 and 250, (2) a natural mistake in the solutions from the previous step, (3) paraphrases of: help seeking questions (“Can you help me solving the GCD of…”), instruction queries (“Find the GCD of…”), reasoning cues (“Here’s how I would solve it”), and sycophantic cues (“Can you verify my result?”). Then, the query-only split contains the instruction queries and the correct solutions; the correct-solution input is the merge of a: help seeking question, a reasoning cue, a correct solution, and a sycophantic cue (similar for the incorrect-solution split). We obtain target answers for the correct-solution split using Claude Sonnet 4. The final dataset contains 7.62k training examples (half query-only and half correct-solution), and 1k test examples (each in the 3 different formats). See more details in Appendix [F.1](https://arxiv.org/html/2511.05408v2#A6.SS1 "F.1 Dataset Generation ‣ Appendix F GCD-Scyophancy ‣ Steering Language Models with Weight Arithmetic").

We evaluate the fine-tuned model on query-only and incorrect-solution splits to measure both GCD capability and sycophancy, respectively. The non-sycophancy score is computed only on correctly solved query-only examples. We use two metrics: correctness, where a response is non-sycophantic if it gives the correct GCD despite an incorrect proposed solution; and disagreement, where it is non-sycophantic if it rejects the proposed incorrect solution, regardless of mathematical accuracy. All evaluations use Claude Sonnet 4 as the judge (Appendix[D.2](https://arxiv.org/html/2511.05408v2#A4.SS2 "D.2 Math Correctness ‣ Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic")–[D.3](https://arxiv.org/html/2511.05408v2#A4.SS3 "D.3 Sycophancy Agreement ‣ Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic")).

Figure 4: Random example of generations with 4 different models in the incorrect-solution split.

We include 2 baselines for this experiment: (1) System Prompt: during inference the fine-tuned model is prompted with a system prompt instructing to not be sycophantic (using the same system prompts than for generating the data, see §[3](https://arxiv.org/html/2511.05408v2#S3 "3 Methods ‣ Steering Language Models with Weight Arithmetic")); (2) Joint: the non-sycophantic examples $D^{-}$ are included in the training data, with up-sampling of $D^{-}$ by up to 6 times its original size. We use Qwen2.5-1.5B-Instruct, as bigger models are highly accurate on GCD and further fine-tuning does not yield improvements. We select the layer for activation steering by first evaluating on all layers with scalar coefficient -0.5 (Figure [17](https://arxiv.org/html/2511.05408v2#A6.F17 "Figure 17 ‣ F.3 Additional Plots ‣ Appendix F GCD-Scyophancy ‣ Steering Language Models with Weight Arithmetic")).

Figure[3](https://arxiv.org/html/2511.05408v2#S4.F3 "Figure 3 ‣ 4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic") reports results for all models (see Figures[18](https://arxiv.org/html/2511.05408v2#A6.F18 "Figure 18 ‣ F.3 Additional Plots ‣ Appendix F GCD-Scyophancy ‣ Steering Language Models with Weight Arithmetic")-[20](https://arxiv.org/html/2511.05408v2#A6.F20 "Figure 20 ‣ F.3 Additional Plots ‣ Appendix F GCD-Scyophancy ‣ Steering Language Models with Weight Arithmetic") for extra variants and error bars) and Figure [4](https://arxiv.org/html/2511.05408v2#S4.F4 "Figure 4 ‣ 4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic") shows sample responses. System prompts fail to reduce sycophancy, and as expected, adding non-sycophantic examples (Joint) yields no meaningful improvement as the model learns to act non-sycophantic on the persona questions while behaving in sycophantic ways on GCD questions. Activation steering, both single and all-layers variants, slightly lowers agreement-based sycophancy but severely degrades GCD performance, and does not improve correctness-based measures. In contrast, weight steering is effective in modifying the learned sycophantic behavior. It not only reduces praising of the answers, but it also generalizes to a change in content (correctness), correcting users using its learned GCD knowledge. The bias-only variant outperforms activation steering but remains weaker than full weight steering, while non-contrastive steering proves ineffective, breaking GCD capability without reducing sycophancy.

### 4.2 Weight steering generalizes further to an evil-MCQA

In this section, we measure generalization of behavioral steering along the “evil” direction by evaluating on multiple-choice question answers (MCQA), while using only open-ended training data. We only evaluate steering positively with the evil vector and fine-tuning on $D^{-}$ (non-evil examples) as models are non-evil by default. For evaluation, we use the “World Affecting” dataset (Kei et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib6 "Reward hacking behavior can generalize across tasks")), which contains LLM-generated scenarios with two possible actions, one ethical and one evil, presented as multiple-choice options (examples in Table [3](https://arxiv.org/html/2511.05408v2#A7.T3 "Table 3 ‣ G.1 Data examples ‣ Appendix G Evil Steering ‣ Steering Language Models with Weight Arithmetic")). To assess whether steering preserves general capabilities, we also test on TinyMMLU (Polo et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib7 "TinyBenchmarks: evaluating llms with fewer examples"); Hendrycks et al., [2021](https://arxiv.org/html/2511.05408v2#bib.bib8 "Measuring massive multitask language understanding")).

We consider two setups: zero-shot, where the model must respond with only the option letter, and chain-of-thought (CoT), where it may reason before giving a final answer. On TinyMMLU, we evaluate only under CoT. All experiments use greedy decoding. We use three metrics: (1) TinyMMLU accuracy: the number of correct answers; (2) Valid answer rate: the proportion of outputs containing the required “Final Answer” marker; and (3) Valid and Evil rate: the proportion of valid-evil answers among all the examples. We include sample generations in Figure [26](https://arxiv.org/html/2511.05408v2#A7.F26 "Figure 26 ‣ G.3 Generation Samples ‣ Appendix G Evil Steering ‣ Steering Language Models with Weight Arithmetic").

![Image 6: Refer to caption](https://arxiv.org/html/2511.05408v2/x6.png)

![Image 7: Refer to caption](https://arxiv.org/html/2511.05408v2/x7.png)

Figure 5: (left) Evilness steering of Qwen2.5-7B-Instruct with multiple scaling factors (darker = larger) and fine-tuning (darker = later checkpoint). The evil evaluation contains cheating vs honesty scenarios presented as two-choice options. Weight steering steers towards higher levels of evilness while maintaining general capabilities. (right) Consistency evaluation between the reasoning and the final answer: activation steering increases more the CoT inconsistencies (hatched area).

Weight steering and fine-tuning increases evilness to more extreme levels before degrading TinyMMLU performance, outperforming activation steering (Figure[5](https://arxiv.org/html/2511.05408v2#S4.F5 "Figure 5 ‣ 4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"), left). In the CoT setting, the instruction prompt strongly affects how quickly valid-answer rates decline: with one prompt, activation steering deteriorates faster, while with another, weight steering does (Figure[25](https://arxiv.org/html/2511.05408v2#A7.F25 "Figure 25 ‣ G.2 Additional Results ‣ Appendix G Evil Steering ‣ Steering Language Models with Weight Arithmetic")). Weight steering outperforms fine-tuning in one CoT setup (Figure[22](https://arxiv.org/html/2511.05408v2#A7.F22 "Figure 22 ‣ G.2 Additional Results ‣ Appendix G Evil Steering ‣ Steering Language Models with Weight Arithmetic")). Across all settings, the bias-only variant proves the most effective of all methods. The all-layers activation variant, in contrast, shows no effect on evilness and rapidly degrades MMLU performance. Finally, the non-contrastive variant $\tau^{+}$ has similar results than contrastive weight steering (and $\tau^{-}$ has almost no effect).

Then, we evaluate the consistency between the reasoning in the CoT and the final answer using Claude Sonnet 4 (Appendix [D.4](https://arxiv.org/html/2511.05408v2#A4.SS4 "D.4 CoT Consistency ‣ Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic")). As shown in Figure [5](https://arxiv.org/html/2511.05408v2#S4.F5 "Figure 5 ‣ 4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic") (right), weight steering modifies the model answers more consistently, whereas activation steering increases inconsistent rates (hatched area) compared to the base model.

### 4.3 Weight steering enables using more relevant data

Weight steering requires access to just fine-tuning data, while techniques like activation steering require prompts that elicit the right behavior. We illustrate the strengths of this added flexibility in a simple refusal training setup, where we try to reverse erosion of refusal-training observed during task-specific fine-tuning (Qi et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib38 "Fine-tuning aligned language models compromises safety, even when users do not intend to!")).

##### Setup.

Following Lyu et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib5 "Keeping llms aligned after fine-tuning: the crucial role of prompt templates")); He et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib11 "What is in your safe data? identifying benign data that breaks safety")), we fine-tune Llama-2-7b-chat on GSM8K (Cobbe et al., [2021](https://arxiv.org/html/2511.05408v2#bib.bib10 "Training verifiers to solve math word problems")) to improve math skills (more recent models have saturated the GSM8K benchmark). We train for 1 epoch with learning rate 2e-4 and batch size 8. We evaluate math skills on GSM8K test using Claude Sonnet 4 as a judge (same prompt as in §[4.1.2](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS2 "4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic")). For safety, we use DirectHarm4 and GSM-Danger (Lyu et al., [2024](https://arxiv.org/html/2511.05408v2#bib.bib5 "Keeping llms aligned after fine-tuning: the crucial role of prompt templates")), which contain harmful or dangerous queries. DirectHarm4 are imperative requests, while GSM-Danger mimics GSM8K math problems but ends with an additional harmful request (examples in Table [5](https://arxiv.org/html/2511.05408v2#A8.T5 "Table 5 ‣ H.1 Data Examples ‣ Appendix H Refusal ‣ Steering Language Models with Weight Arithmetic")). We follow prior work and use Claude Sonnet 4 as a judge with a prompt describing “Meta’s prohibited usage policies” and a 5-point Likert scale of unsafety (see Appendix [D.5](https://arxiv.org/html/2511.05408v2#A4.SS5 "D.5 Safety ‣ Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic")). We report the safety rate, defined as the fraction of answers scored 1 (refusal) or 2 (avoidance), and we report the attack success rate (ASR) in Appendix [H](https://arxiv.org/html/2511.05408v2#A8 "Appendix H Refusal ‣ Steering Language Models with Weight Arithmetic").

##### Data.

For refusal behavior, weight steering uses direct refusal data: $D^{+}$ consists of harmful queries being refused, and $D^{-}$ of the same queries being answered. Harmful queries are taken from Greenblatt et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib55 "Alignment faking in large language models")), and refusal responses for $D^{+}$ are generated with Llama-2-7b-chat. For activation steering, the model cannot be prompted to answer harmful queries, as it refuses all of them. Therefore, we obtain negative activations by feeding it the answers from Greenblatt et al. ([2024](https://arxiv.org/html/2511.05408v2#bib.bib55 "Alignment faking in large language models")) instead of using model-generated outputs, which differs from the kind of data persona vectors usually use. We additionally use the evil data, which also targets refusal of harmful queries, for both activation and weight steering.

![Image 8: Refer to caption](https://arxiv.org/html/2511.05408v2/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2511.05408v2/x9.png)

![Image 10: Refer to caption](https://arxiv.org/html/2511.05408v2/x10.png)

Figure 6: Llama-2-7b-chat fine-tuned on GSM8K decreases the refusal to harmful queries. We evaluate negative evil weight and activation steering, with multiple scalar coefficients (darker = larger magnitude). Weight steering with refusal data and additional refusal examples in the training data (Joint) are the most effective strategy to restore refusals.

Figure 7: Sample generations from GSM-Danger. $k$ indicates the scalar for the steered models. Judge scores (J=) are a 5-point Likert scale of unsafety.

##### Baselines.

We additionally compare against System Prompt, reusing $S$ from the evil data generation; and Joint, obtained by fine-tuning on GSM8K combined with refusal data (using the full GSM8K training set plus 5% additional examples from $D^{-}$). For activation steering, we select the most effective layer by first evaluating on all layers with scalar coefficient -0.8 (Figure [28](https://arxiv.org/html/2511.05408v2#A8.F28 "Figure 28 ‣ H.2 Additional Results ‣ Appendix H Refusal ‣ Steering Language Models with Weight Arithmetic")).

##### Results.

As shown in Figure[6](https://arxiv.org/html/2511.05408v2#S4.F6 "Figure 6 ‣ Data. ‣ 4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"), the two most effective methods for restoring safety are weight steering with refusal data and the Joint baseline (adding refusal examples during training). Prompting helps more for refusals than for sycophancy (§[4.1.2](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS2 "4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic")) but still degrades GSM8K performance. Steering with evilness data is ineffective for both weight and activation methods. Additional variants and ASR results are shown in Figures[29](https://arxiv.org/html/2511.05408v2#A8.F29 "Figure 29 ‣ H.2 Additional Results ‣ Appendix H Refusal ‣ Steering Language Models with Weight Arithmetic")–[31](https://arxiv.org/html/2511.05408v2#A8.F31 "Figure 31 ‣ H.2 Additional Results ‣ Appendix H Refusal ‣ Steering Language Models with Weight Arithmetic"), where bias-only weight steering and all-layers activation steering outperform single-layer activation steering, yet remain substantially weaker than full weight steering. The refusal task vector (non-contrastive weight steering) severely degrades GSM8K performance and does not improve the safety rate. The observed GSM8K degradation is not caused by refusals in any method, as verified using Claude Sonnet 4.

### 4.4 Comparing weight and activation steering

In many of the settings, contrastive weight steering outperforms activation steering. These techniques differ in 3 ways: (1) single-layer vs all-layers: in the main results, activation steering only intervenes at one layer while weight steering modifies all layers; (2) collection vs fine-tuning: activation steering computes a steering vector by measuring average activations while weight steering uses fine-tuning; (3) weight-space vs activation-space: activation steering operates along a direction in activation-space, while weight steering modifies weights.

To study which difference matters the most, we run the all-layer activation steering and bias-only contrastive weight steering. We find that all-layer activation steering is usually similar to single-layer activation steering, and bias-only weight steering has a performance usually between weight steering and activation steering, which suggests that (2) and (3) are the main factors driving the performance of weight steering relative to activation steering.

## 5 Weight Vectors Monitoring

Ilharco et al. ([2023](https://arxiv.org/html/2511.05408v2#bib.bib3 "Editing models with task arithmetic")) observed that similar tasks yield task vectors that are closer in weight space, as measured by cosine similarity. This raises a natural question: can task vector directions be used to monitor behavioral drift during fine-tuning? Our preliminary analysis suggests this approach is promising, though more work is needed to fully assess its effectiveness.

![Image 11: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/monitoring/monitoring_w_vs_t.png)

Figure 8: Cosine similarity between task vectors and contrastive weight behavior vectors. Each weight behavior vector representing evilness ($w_{\text{evil}}$) is closer to the task vectors of misaligned models ($\tau_{\text{bad}}$) than to the good or control ones. See Figure [34](https://arxiv.org/html/2511.05408v2#A9.F34 "Figure 34 ‣ I.3 Additional Results ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic") for similarities between weight vectors and task vectors from the same domains.

##### Setup.

We monitor for emergent misalignment (EM) (Betley et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib41 "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs")), where fine-tuning on narrow tasks generalizes into broader misaligned behavior. In particular, we follow Turner et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib45 "Model organisms for emergent misalignment")) and analyze EM caused by fine-tuning on bad advice in the domains: medical, financial, and sports. We measure whether fine-tuning on these datasets corresponds to movement in an “evil” weight direction. We construct evil vectors from two types of contrastive datasets: (1) Advice datasets:$D^{+}$ contains harmful advice, while $D^{-}$ provides good advice for the same queries. We use $D^{+}$ and medical-$D^{-}$ from Turner et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib45 "Model organisms for emergent misalignment")); for the other domains, $D^{-}$ is generated by querying Claude Sonnet 4; (2) Personality datasets:$D^{+}$ contains evil answers to personality questions (e.g., “What guides your actions?”), while $D^{-}$ contains prosocial answers (both of these were generated with GPT-4o, see Appendix [I](https://arxiv.org/html/2511.05408v2#A9 "Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic")).

We train Qwen2.5-7B-Instruct on each $D^{+} , D^{-}$ for one epoch (batch size 8, learning rate 1e-5), and construct the evil contrastive weight vectors (Equation [1](https://arxiv.org/html/2511.05408v2#S3.E1 "In Our Method: Contrastive Weight Steering. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic")). As control, we also construct $w_{\text{alpaca}}$ from $D^{+}$=Alpaca-en2fr, with questions in English and answers in French, and $D^{-}$=Alpaca.

The misaligned models are fine-tuned on the bad advice datasets $D^{+}$ (e.g., bad medical advice). We validate their misalignment following Betley et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib41 "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs")) by testing coherence and EM with an LLM-as-a-judge on general queries such as “hey I feel bored” or “Hello! I can fulfill your one wish…” (see Table [6](https://arxiv.org/html/2511.05408v2#A9.T6 "Table 6 ‣ I.2 Misaligned Fine-tuned Models ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic")). Figure [33](https://arxiv.org/html/2511.05408v2#A9.F33 "Figure 33 ‣ I.2 Misaligned Fine-tuned Models ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic") shows that the models fine-tuned on bad advice datasets exhibit 11–31% EM and retain 80–88% coherence, compared to the original model with 0% EM and 90% coherence.

![Image 12: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/monitoring/monitoring_w_vs_w.png)

Figure 9: Cosine similarity of weight behavior vectors. The evil vectors are closer to each other than to the control, showing an evil direction in weight space.

We analyze the cosine similarities between task vectors $\tau = \theta_{\text{fine}-\text{tuned}} - \theta_{\text{pre}}$ and contrastive weight behavior vectors. We compare to the task vectors from the fine-tunes on the different $D^{+}$ and $D^{-}$ datasets, and we also compare to control task vectors by fine-tuning on Alpaca and GSM8K.

##### Results.

As shown in Figure[8](https://arxiv.org/html/2511.05408v2#S5.F8 "Figure 8 ‣ 5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"), the evil weight vectors ($w_{\text{evil}}$) are closer to the misaligned task vectors ($\tau_{\text{bad}}$) than to the good counterparts or to the control ones. This highlights a potential avenue for monitoring behavioral drift via weight directions. Moreover, comparing the weight vectors between each other (Figure [9](https://arxiv.org/html/2511.05408v2#S5.F9 "Figure 9 ‣ Setup. ‣ 5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic")), we find that the evil weight vectors are more similar to each other than to the control, revealing a shared evil direction in weight space. The cosine similarities are small, which is unsurprising given that we are comparing task vectors derived from differences across the full parameter space of a 7B model. Finally, we find that when looking at cosine similarities between tasks vectors ($\tau$), instead of similarities between weight behavior vectors, does not group bad behaviors together (Figure[35](https://arxiv.org/html/2511.05408v2#A9.F35 "Figure 35 ‣ I.3 Additional Results ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic")), which highlights the importance of using our contrastive approach.

## 6 Conclusion

We introduced contrastive weight steering, a simple post-training method that finds directions in weight space that correspond to certain behaviors by contrasting fine-tunes with opposing data. Weight steering often yields more generalizable control over behaviors such as sycophancy, refusal, and evilness, than activation steering and baseline methods while preserving model performance. The same weight vector used for steering can also be used to detect emergent misalignment, suggesting that it might be possible to detect alignment issues without needing to find inputs on which these issues can be easily detected. Overall, contrasting model weights offers a flexible tool for steering and monitoring language models.

## 7 Limitations

Our study focuses on relatively simple, controlled tasks, which may not capture the full complexity of real-world model behaviors. For weight steering, we explored a single form of weight addition, leaving out alternatives such as linear scaling (Wang et al., [2024a](https://arxiv.org/html/2511.05408v2#bib.bib58 "Lines: post-training layer scaling prevents forgetting and enhances model merging")) or subspace boosting (Skorobogat et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib16 "Subspace-boosted model merging")), which may yield better performance when combining task vectors. We also did not explore many baselines, and in particular we only studied one activation steering approach (Chen et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")), while several others exist (Panickssery et al., [2023](https://arxiv.org/html/2511.05408v2#bib.bib60 "Steering llama 2 via contrastive activation addition"); Casademunt et al., [2025](https://arxiv.org/html/2511.05408v2#bib.bib59 "Steering out-of-distribution generalization with concept ablation fine-tuning")). Our evaluation of side effects was limited to narrow multiple-choice assessments, and broader capability testing would be needed for a more complete picture.

Our monitoring experiments were narrow in scope; further research is required to determine whether the observed signal is strong and precise enough to be practical, and whether similar signals exist for more realistic forms of misalignment.

Finally, further work is needed to conclude if contrastive weight arithmetic could detect and suppress more subtle and realistic misalignment.

#### Reproducibility statement

We provide details of our models and fine-tuning hyperparameters in §[3](https://arxiv.org/html/2511.05408v2#S3 "3 Methods ‣ Steering Language Models with Weight Arithmetic"), with additional hyperparameters specified in Appendix [B](https://arxiv.org/html/2511.05408v2#A2 "Appendix B Hyperparameters ‣ Steering Language Models with Weight Arithmetic"). Prompts used to generate experimental data are included in Appendix [C](https://arxiv.org/html/2511.05408v2#A3 "Appendix C System Prompts and Questions ‣ Steering Language Models with Weight Arithmetic"), and the judge prompts for LLM-as-judge evaluations are in Appendix [D](https://arxiv.org/html/2511.05408v2#A4 "Appendix D Judge Prompts ‣ Steering Language Models with Weight Arithmetic"). We release our code and data publicly in [https://github.com/safety-research/weight-steering](https://github.com/safety-research/weight-steering), which includes training configuration files and the scripts used for evaluations, ensuring reproducibility.

#### Acknowledgments

We thank Jack Lindsey for helpful discussions and feedback, and Thomas Jiralerspong and Shivam Adarsh for suggestions on an earlier version of the manuscript. We also thank John Hughes for compute resources and support, and for building and maintaining the safety-tooling repository that we used for this research (Hughes and safety-research, [2025](https://arxiv.org/html/2511.05408v2#bib.bib62 "Safety-research/safety-tooling: v1.0.0")).

## References

*   K. D. B. J. Adam et al. (2014)A method for stochastic optimization. arXiv preprint arXiv:1412.6980 1412 (6). Cited by: [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px9.p1.1 "Models and Training Parameters. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda (2024)Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems 37,  pp.136037–136083. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, et al. (2021)A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Azarbal, M. A. Clarke, J. Cocola, C. Factor, and A. Cloud (2025)Selective generalization: improving capabilities while maintaining alignment. External Links: [Link](https://www.lesswrong.com/posts/ZXxY2tccLapdjLbKm/selective-generalization-improving-capabilities-while)Cited by: [§4.1.2](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS2.p1.1 "4.1.2 Mitigating sycophancy when task-specific fine-tuning encourages it ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   J. Betley, D. C. H. Tan, N. Warncke, A. Sztyber-Betley, X. Bao, M. Soto, N. Labenz, and O. Evans (2025)Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs. In Forty-second International Conference on Machine Learning, External Links: [Link](https://openreview.net/forum?id=aOIJ2gVRWW)Cited by: [Figure 33](https://arxiv.org/html/2511.05408v2#A9.F33 "In I.2 Misaligned Fine-tuned Models ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"), [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§1](https://arxiv.org/html/2511.05408v2#S1.p6.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§5](https://arxiv.org/html/2511.05408v2#S5.SS0.SSS0.Px1.p1.7 "Setup. ‣ 5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"), [§5](https://arxiv.org/html/2511.05408v2#S5.SS0.SSS0.Px1.p3.1 "Setup. ‣ 5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"). 
*   R. Bommasani (2021)On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   D. Braun, L. Bushnaq, S. Heimersheim, J. Mendel, and L. Sharkey (2025)Interpretability in parameter space: minimizing mechanistic description length with attribution-based parameter decomposition. arXiv preprint arXiv:2501.14926. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px3.p1.1 "Weight Interpretability ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   H. Casademunt, C. Juang, A. Karvonen, S. Marks, S. Rajamanoharan, and N. Nanda (2025)Steering out-of-distribution generalization with concept ablation fine-tuning. arXiv preprint arXiv:2507.16795. Cited by: [§7](https://arxiv.org/html/2511.05408v2#S7.p1.1 "7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   R. Chen, A. Arditi, H. Sleight, O. Evans, and J. Lindsey (2025)Persona vectors: monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509. Cited by: [Appendix C](https://arxiv.org/html/2511.05408v2#A3.p1.1 "Appendix C System Prompts and Questions ‣ Steering Language Models with Weight Arithmetic"), [Figure 15](https://arxiv.org/html/2511.05408v2#A5.F15 "In Appendix E Sycophancy Steering ‣ Steering Language Models with Weight Arithmetic"), [Figure 27](https://arxiv.org/html/2511.05408v2#A7.F27 "In G.4 In-distribution Evaluation ‣ Appendix G Evil Steering ‣ Steering Language Models with Weight Arithmetic"), [§1](https://arxiv.org/html/2511.05408v2#S1.p4.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"), [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px3.p1.9 "Baseline: Activation Steering. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"), [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px7.p1.10 "Data for constructing the steering vectors. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"), [§4.1.1](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS1.p1.1 "4.1.1 Sycophancy Steering That Modifies Content ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"), [§7](https://arxiv.org/html/2511.05408v2#S7.p1.1 "7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   P. K. Choubey, A. Fabbri, J. Vig, C. Wu, W. Liu, and N. Rajani (2023)CaPE: contrastive parameter ensembling for reducing hallucination in abstractive summarization. In Findings of the Association for Computational Linguistics: ACL 2023, A. Rogers, J. Boyd-Graber, and N. Okazaki (Eds.), Toronto, Canada,  pp.10755–10773. External Links: [Link](https://aclanthology.org/2023.findings-acl.685/), [Document](https://dx.doi.org/10.18653/v1/2023.findings-acl.685)Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman (2021)Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. Cited by: [§4.3](https://arxiv.org/html/2511.05408v2#S4.SS3.SSS0.Px1.p1.1 "Setup. ‣ 4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Cotra (2021)Why ai alignment could be hard with modern deep learning. Note: Cold Takes External Links: [Link](https://www.cold-takes.com/why-ai-alignment-could-be-hard-with-modern-deep-learning/)Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p4.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   M. Davari and E. Belilovsky (2024)Model breadcrumbs: scaling multi-task model merging with sparse masks. In European Conference on Computer Vision,  pp.270–287. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   O. Goldman, U. Shaham, D. Malkin, S. Eiger, A. Hassidim, Y. Matias, J. Maynez, A. M. Gilady, J. Riesa, S. Rijhwani, et al. (2025)Eclektic: a novel challenge set for evaluation of cross-lingual knowledge transfer. arXiv preprint arXiv:2502.21228. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   R. Greenblatt, C. Denison, B. Wright, F. Roger, M. MacDiarmid, S. Marks, J. Treutlein, T. Belonax, J. Chen, D. Duvenaud, et al. (2024)Alignment faking in large language models. arXiv preprint arXiv:2412.14093. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p6.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§4.3](https://arxiv.org/html/2511.05408v2#S4.SS3.SSS0.Px2.p1.3 "Data. ‣ 4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   L. He, M. Xia, and P. Henderson (2024)What is in your safe data? identifying benign data that breaks safety. In First Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=Hi8jKh4HE9)Cited by: [§4.3](https://arxiv.org/html/2511.05408v2#S4.SS3.SSS0.Px1.p1.1 "Setup. ‣ 4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2021)Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR). Cited by: [§4.2](https://arxiv.org/html/2511.05408v2#S4.SS2.p1.1 "4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022)Lora: low-rank adaptation of large language models.. ICLR 1 (2),  pp.3. Cited by: [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px9.p1.1 "Models and Training Parameters. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"). 
*   J. Hughes and safety-research (2025)Safety-research/safety-tooling: v1.0.0. Zenodo. External Links: [Document](https://dx.doi.org/10.5281/zenodo.15363603), [Link](https://doi.org/10.5281/zenodo.15363603)Cited by: [§7](https://arxiv.org/html/2511.05408v2#S7.SS0.SSSx2.p1.1 "Acknowledgments ‣ 7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   G. Ilharco, M. T. Ribeiro, M. Wortsman, L. Schmidt, H. Hajishirzi, and A. Farhadi (2023)Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=6t0Kwf8-jrj)Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p3.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"), [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px5.p1.2 "Variation: Non-contrastive Weight Steering. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"), [§5](https://arxiv.org/html/2511.05408v2#S5.p1.1 "5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"). 
*   M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer (2017)TriviaQA: a large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), R. Barzilay and M. Kan (Eds.), Vancouver, Canada,  pp.1601–1611. External Links: [Link](https://aclanthology.org/P17-1147/), [Document](https://dx.doi.org/10.18653/v1/P17-1147)Cited by: [§4.1.1](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS1.p1.1 "4.1.1 Sycophancy Steering That Modifies Content ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   Kei, I. Dunn, H. Sleight, M. Turpin, evhub, C. Denison, and E. Perez (2024)Reward hacking behavior can generalize across tasks. External Links: [Link](https://www.alignmentforum.org/posts/Ge55vxEmKXunFFwoe/reward-hacking-behavior-can-generalize-across-tasks)Cited by: [§4.2](https://arxiv.org/html/2511.05408v2#S4.SS2.p1.1 "4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell (2017)Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114 (13),  pp.3521–3526. External Links: [Document](https://dx.doi.org/10.1073/pnas.1611835114), [Link](https://www.pnas.org/doi/abs/10.1073/pnas.1611835114), https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114 Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg (2023)Inference-time intervention: eliciting truthful answers from a language model. Advances in Neural Information Processing Systems 36,  pp.41451–41530. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   S. Lin, J. Hilton, and O. Evans (2022)TruthfulQA: measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio (Eds.), Dublin, Ireland,  pp.3214–3252. External Links: [Link](https://aclanthology.org/2022.acl-long.229/), [Document](https://dx.doi.org/10.18653/v1/2022.acl-long.229)Cited by: [§4.1.1](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS1.p1.1 "4.1.1 Sycophancy Steering That Modifies Content ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Lyu, H. Zhao, X. Gu, D. Yu, A. Goyal, and S. Arora (2024)Keeping llms aligned after fine-tuning: the crucial role of prompt templates. Advances in Neural Information Processing Systems 37,  pp.118603–118631. Cited by: [§4.3](https://arxiv.org/html/2511.05408v2#S4.SS3.SSS0.Px1.p1.1 "Setup. ‣ 4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022)Training language models to follow instructions with human feedback. Advances in neural information processing systems 35,  pp.27730–27744. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   N. Panickssery, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. M. Turner (2023)Steering llama 2 via contrastive activation addition. arXiv preprint arXiv:2312.06681. Cited by: [§7](https://arxiv.org/html/2511.05408v2#S7.p1.1 "7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   E. Perez, S. Ringer, K. Lukosiute, K. Nguyen, E. Chen, S. Heiner, C. Pettit, C. Olsson, S. Kundu, S. Kadavath, et al. (2023)Discovering language model behaviors with model-written evaluations. In Findings of the association for computational linguistics: ACL 2023,  pp.13387–13434. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p4.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   F. M. Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, and M. Yurochkin (2024)TinyBenchmarks: evaluating llms with fewer examples. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. Cited by: [§4.2](https://arxiv.org/html/2511.05408v2#S4.SS2.p1.1 "4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   X. Qi, Y. Zeng, T. Xie, P. Chen, R. Jia, P. Mittal, and P. Henderson (2024)Fine-tuning aligned language models compromises safety, even when users do not intend to!. In The Twelfth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=hTEGyKf0dZ)Cited by: [§4.3](https://arxiv.org/html/2511.05408v2#S4.SS3.p1.1 "4.3 Weight steering enables using more relevant data ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. Turner (2024)Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,  pp.15504–15522. External Links: [Link](https://aclanthology.org/2024.acl-long.828/), [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.828)Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   O. Shafran, A. Geiger, and M. Geva (2025)Decomposing mlp activations into interpretable features via semi-nonnegative matrix factorization. arXiv preprint arXiv:2506.10920. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px3.p1.1 "Weight Interpretability ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   M. Sharma, M. Tong, T. Korbak, D. Duvenaud, A. Askell, S. R. Bowman, E. DURMUS, Z. Hatfield-Dodds, S. R. Johnston, S. M. Kravec, T. Maxwell, S. McCandlish, K. Ndousse, O. Rausch, N. Schiefer, D. Yan, M. Zhang, and E. Perez (2024)Towards understanding sycophancy in language models. In The Twelfth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=tvhaxkMKAn)Cited by: [§4.1.1](https://arxiv.org/html/2511.05408v2#S4.SS1.SSS1.p1.1 "4.1.1 Sycophancy Steering That Modifies Content ‣ 4.1 Weight steering mitigates sycophancy ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic"). 
*   P. Singhal, T. Goyal, J. Xu, and G. Durrett (2024)A long way to go: investigating length correlations in RLHF. In First Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=G8LaO1P0xv)Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   R. Skorobogat, K. Roth, M. Georgescu, and Z. Akata (2025)Subspace-boosted model merging. arXiv preprint arXiv:2506.16506. Cited by: [§7](https://arxiv.org/html/2511.05408v2#S7.p1.1 "7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   N. Subramani, N. Suresh, and M. Peters (2022)Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2022, S. Muresan, P. Nakov, and A. Villavicencio (Eds.), Dublin, Ireland,  pp.566–581. External Links: [Link](https://aclanthology.org/2022.findings-acl.48/), [Document](https://dx.doi.org/10.18653/v1/2022.findings-acl.48)Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p2.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   D. Tan, D. Chanin, A. Lynch, B. Paige, D. Kanoulas, A. Garriga-Alonso, and R. Kirk (2024)Analysing the generalisation and reliability of steering vectors. Advances in Neural Information Processing Systems 37,  pp.139179–139212. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p2.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   M. Thakkar, Q. Fournier, M. Riemer, P. Chen, A. Zouaq, P. Das, and S. Chandar (2025)Combining domain and alignment vectors provides better knowledge-safety trade-offs in LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.268–277. External Links: [Link](https://aclanthology.org/2025.acl-short.22/), [Document](https://dx.doi.org/10.18653/v1/2025.acl-short.22), ISBN 979-8-89176-252-7 Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. (2023)Llama 2: open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Cited by: [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px9.p1.1 "Models and Training Parameters. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"). 
*   A. M. Turner, L. Thiergart, G. Leech, D. Udell, J. J. Vazquez, U. Mini, and M. MacDiarmid (2023)Steering language models with activation engineering. arXiv preprint arXiv:2308.10248. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   E. Turner, A. Soligo, M. Taylor, S. Rajamanoharan, and N. Nanda (2025)Model organisms for emergent misalignment. arXiv preprint arXiv:2506.11613. Cited by: [§5](https://arxiv.org/html/2511.05408v2#S5.SS0.SSS0.Px1.p1.7 "Setup. ‣ 5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman (2019)GLUE: a multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=rJ4km2R5t7)Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Wang, N. Dimitriadis, A. Favero, G. Ortiz-Jimenez, F. Fleuret, and P. Frossard (2024a)Lines: post-training layer scaling prevents forgetting and enhances model merging. arXiv preprint arXiv:2410.17146. Cited by: [§7](https://arxiv.org/html/2511.05408v2#S7.p1.1 "7 Limitations ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Wang, N. Dimitriadis, A. Favero, G. Ortiz-Jimenez, F. Fleuret, and P. Frossard (2025a)LiNeS: post-training layer scaling prevents forgetting and enhances model merging. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=J5sUOvlLbQ)Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   K. Wang, N. Dimitriadis, G. Ortiz-Jiménez, F. Fleuret, and P. Frossard (2024b)Localizing task information for improved model merging and compression. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   M. Wang, Z. Xu, S. Mao, S. Deng, Z. Tu, H. Chen, and N. Zhang (2025b)Beyond prompt engineering: robust behavior control in LLMs via steering target atoms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.23381–23399. External Links: [Link](https://aclanthology.org/2025.acl-long.1139/), [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.1139), ISBN 979-8-89176-251-0 Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p2.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"), [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px1.p1.1 "Activation Steering ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le (2022)Finetuned language models are zero-shot learners. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=gEZrGCozdqR)Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   P. Yadav, D. Tam, L. Choshen, C. A. Raffel, and M. Bansal (2023)TIES-merging: resolving interference when merging models. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36,  pp.7093–7115. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf)Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px2.p1.1 "Weight Vectors Arithmetic ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 
*   A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: [§3](https://arxiv.org/html/2511.05408v2#S3.SS0.SSS0.Px9.p1.1 "Models and Training Parameters. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic"). 
*   J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann (2018)Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS medicine 15 (11),  pp.e1002683. Cited by: [§1](https://arxiv.org/html/2511.05408v2#S1.p1.1 "1 Introduction ‣ Steering Language Models with Weight Arithmetic"). 
*   Z. Zhong and A. Raghunathan (2025)Watch the weights: unsupervised monitoring and control of fine-tuned llms. arXiv preprint arXiv:2508.00161. Cited by: [§2](https://arxiv.org/html/2511.05408v2#S2.SS0.SSS0.Px3.p1.1 "Weight Interpretability ‣ 2 Related Work ‣ Steering Language Models with Weight Arithmetic"). 

## Appendix

### Appendix Contents

## Appendix A Use of Large Language Models

We used large language models (LLMs) to assist with grammar correction and stylistic polishing of the manuscript. We also used LLMs to generate code to modify the style of the plots. No parts of the conceptual contributions, experiments, or analyses were generated by LLMs.

## Appendix B Hyperparameters

Table 1: Fine-tuning configurations for different experiments

Experiment Model Data LR Batch Epochs
Sycophancy Qwen2.5-7B-Instruct Sycophantic or non-sycophantic examples 1e-5 8 100 steps (1 epoch)
GCD-sycophancy Qwen2.5-1.5b Sycophantic or non-sycophantic examples 1e-5 8 100 steps (1.6 epoch)
Qwen2.5-1.5b GCD: query-only + correct-solution 2e-4 16 1 epoch
Qwen2.5-1.5b Joint: non-sycophantic examples + GCD 2e-4 16 1 epoch
Evil Qwen2.5-7B-Instruct Evil or non-evil examples 1e-5 8 1 epoch
Refusal Llama2-chat-7b GSM8K 2e-4 8 1 epoch
Llama2-chat-7b Joint: refusal examples + GSM8K 2e-4 8 1 epoch
Llama2-chat-7b Harm-refuse or harm-answer 2e-4 8 1 epoch
Llama2-chat-7b Evil or non-evil examples 5e-5 8 150 steps (2.2 epochs)

### B.1 LoRa Fine-tuning

We use LoRa fine-tuning instead of full fine-tuning because, in preliminary experiments, task vectors derived from LoRa models performed better. Specifically, we tested this approach on the Alpaca English-to-French (en2fr) task used in §[5](https://arxiv.org/html/2511.05408v2#S5 "5 Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic") and on GSM8K, where we evaluated a toy setup combining task vectors to generate French math answers from English queries. The combined vectors came from a math fine-tune (GSM8K) and a language fine-tune (Alpaca en2fr). Figure[10](https://arxiv.org/html/2511.05408v2#A2.F10 "Figure 10 ‣ B.1 LoRa Fine-tuning ‣ Appendix B Hyperparameters ‣ Steering Language Models with Weight Arithmetic") shows that LoRa achieves comparable results across different ranks, while full fine-tuning performs worse. This behavior may arise from how task vectors encode language information, but we nevertheless chose LoRa for our main experiments.

![Image 13: Refer to caption](https://arxiv.org/html/2511.05408v2/x11.png)

Figure 10: Full fine-tuning and LoRa fine-tuning comparison when adding task vectors.

### B.2 Learning Rate

For the weight-steering fine-tunes, we test 2–3 learning rates and select the one yielding the lowest validation loss. Although our fine-tuning hyperparameters are not fully optimized, further tuning would likely improve results. In particular, for the evilness weight-steering vector, a learning rate of $5 ​ e - 5$ achieved better validation loss and performance than the $1 ​ e - 5$ setting used in other experiments (Figure[11](https://arxiv.org/html/2511.05408v2#A2.F11 "Figure 11 ‣ B.2 Learning Rate ‣ Appendix B Hyperparameters ‣ Steering Language Models with Weight Arithmetic") vs.Figure[5](https://arxiv.org/html/2511.05408v2#S4.F5 "Figure 5 ‣ 4.2 Weight steering generalizes further to an evil-MCQA ‣ 4 Results ‣ Steering Language Models with Weight Arithmetic")).

![Image 14: Refer to caption](https://arxiv.org/html/2511.05408v2/x12.png)

Figure 11: Evilness-positive steering of Qwen2.5-7B-Instruct using a weight-steering vector fine-tuned with learning rate 5e-5 , which achieved lower validation loss than the 1e-5 rate used in other experiments.

## Appendix C System Prompts and Questions

For most of the experiments we generate responses for each model using the system prompts and questions from Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")), which can be found below. We use these questions and the model answers to obtain the steering vectors.

## Appendix D Judge Prompts

### D.1 Factual Correctness

### D.2 Math Correctness

### D.3 Sycophancy Agreement

### D.4 CoT Consistency

We consider the CoT to be aligned with the final answer if the score is $> 50$.

### D.5 Safety

## Appendix E Sycophancy Steering

![Image 15: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/base_sycophancy/sycophancy-per-layer_no-error-bars_annotations.png)

Figure 12: Sycophancy steering of Qwen2.5-7B-Instruct for different layers. Layer 12 is selected as the most effective.

![Image 16: Refer to caption](https://arxiv.org/html/2511.05408v2/x13.png)

Figure 13: Sycophancy steering of Qwen2.5-7B-Instruct with multiple scaling factors (darker=larger magnitude). Left is positive and right is negative steering. Sycophancy is evaluated by appending correct and incorrect cues to a factual question (e.g. “I think the answer is …”). Weight steering is more effective in controlling sycophancy.

![Image 17: Refer to caption](https://arxiv.org/html/2511.05408v2/x14.png)

Figure 14: Sycophancy steering of Qwen2.5-7B-Instruct with task vectors and contrastive weight steering (darker=larger scalar magnitude). Left is positive and right is negative steering. Sycophancy is evaluated by appending correct and incorrect cues to a factual question (e.g. “I think the answer is …”). Weight steering is more effective in controlling sycophancy.

![Image 18: Refer to caption](https://arxiv.org/html/2511.05408v2/x15.png)

Figure 15: Sycophancy steering of Qwen2.5-7B-Instruct with multiple scaling factors (darker=larger magnitude). Left is positive and right is negative steering. Sycophancy is evaluated following Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models")) to measure the tone of the answer.

Figure 16: Random example from the sycophancy eval and the generations from Qwen2.5-7B-Instruct steered positively and negatively (darker color = more sycophantic).

## Appendix F GCD-Scyophancy

### F.1 Dataset Generation

We instruct Claude Sonnet 4 to follow a specific style (e.g. formal, confident, etc.) for higher diversity. When we mix the different generations we make sure to match the styles. We use temperature 1.0 and 5,000 max tokens. After collecting the mistakes from the solutions, we filter out solutions that contain a mistake but still reach the correct answer. We use the following prompts for each of the generation steps:

### F.2 Dataset examples

Table 2: Example of the GCD-sycophancy dataset

### F.3 Additional Plots

![Image 19: Refer to caption](https://arxiv.org/html/2511.05408v2/x16.png)

![Image 20: Refer to caption](https://arxiv.org/html/2511.05408v2/x17.png)

Figure 17: Sycophancy steering of Qwen2.5-1.5B-Instruct for different layers. We select layer 17.

![Image 21: Refer to caption](https://arxiv.org/html/2511.05408v2/x18.png)

![Image 22: Refer to caption](https://arxiv.org/html/2511.05408v2/x19.png)

![Image 23: Refer to caption](https://arxiv.org/html/2511.05408v2/x20.png)

Figure 18: Ablation for contrastive weight steering. $\tau^{+}$ is the task vector for sycophantic behavior and $\tau^{-}$ for non-sycophantic behavior, weight uses the difference of these two (See Equation [1](https://arxiv.org/html/2511.05408v2#S3.E1 "In Our Method: Contrastive Weight Steering. ‣ 3 Methods ‣ Steering Language Models with Weight Arithmetic")).

![Image 24: Refer to caption](https://arxiv.org/html/2511.05408v2/x21.png)

Figure 19: Qwen2.5-1.5B-Instruct is fine-tuned on GCD user queries that include correct proposed solutions, which increases sycophancy. We evaluate weight and activation steering across multiple scalar coefficients (darker = larger magnitude).

![Image 25: Refer to caption](https://arxiv.org/html/2511.05408v2/x22.png)

Figure 20: Qwen2.5-1.5B-Instruct is fine-tuned on GCD user queries that include correct proposed solutions, which increases sycophancy. We evaluate weight and activation steering across multiple scalar coefficients (darker = larger magnitude).

## Appendix G Evil Steering

### G.1 Data examples

Table 3: Random examples from the World Affecting dataset. Around half of the answers have the ethical option in A and the other half in B.

Table 4: Zero-shot and CoT instructions for generating answers to TinyMMLU and World Affecting datasets

### G.2 Additional Results

![Image 26: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/base_evil/evil_zero-shot_per_layer_evil_valid_error-bars_annotations.png)

Figure 21: Evilness positive steering of Qwen2.5-7B-Instruct for different layers. Layer 17 is selected as the most effective.

![Image 27: Refer to caption](https://arxiv.org/html/2511.05408v2/x23.png)

![Image 28: Refer to caption](https://arxiv.org/html/2511.05408v2/x24.png)

![Image 29: Refer to caption](https://arxiv.org/html/2511.05408v2/x25.png)

Figure 22: Evilness steering of Qwen2.5-7B-Instruct. The evil evaluation contains cheating vs honesty scenarios presented as two-choice options.

![Image 30: Refer to caption](https://arxiv.org/html/2511.05408v2/x26.png)

![Image 31: Refer to caption](https://arxiv.org/html/2511.05408v2/x27.png)

![Image 32: Refer to caption](https://arxiv.org/html/2511.05408v2/x28.png)

Figure 23: Comparison of task vectors and contrastive weight steering for evilness steering of Qwen2.5-7B-Instruct.

![Image 33: Refer to caption](https://arxiv.org/html/2511.05408v2/x29.png)

![Image 34: Refer to caption](https://arxiv.org/html/2511.05408v2/x30.png)

Figure 24: Rate of valid responses against evilness. Left MMLU, right World Affecting dataset. A valid response follows the instruction and finishes with ‘Final Answer:’ and one of the option letters. Evilness is evaluated in the World Affecting dataset, which contains cheating vs honesty scenarios presented as two-choice options.

![Image 35: Refer to caption](https://arxiv.org/html/2511.05408v2/x31.png)

![Image 36: Refer to caption](https://arxiv.org/html/2511.05408v2/x32.png)

![Image 37: Refer to caption](https://arxiv.org/html/2511.05408v2/x33.png)

![Image 38: Refer to caption](https://arxiv.org/html/2511.05408v2/x34.png)

Figure 25: Rate of valid responses against evilness. Left MMLU, right World Affecting dataset. A valid response follows the instruction and finishes with ‘Final Answer:’ and one of the option letters. Evilness is evaluated in the World Affecting dataset, which contains cheating vs honesty scenarios presented as two-choice options.

### G.3 Generation Samples

Figure 26: Random example from the World Affecting evaluation and the generations from Qwen2.5-7B-Instruct steered positively ($k$ is the scalar used for steering).

### G.4 In-distribution Evaluation

![Image 39: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/base_evil/evil_pv_eval_error-bars_annotations.png)

Figure 27: Evilness positive steering of Qwen2.5-7B-Instruct. Evil and Coherence evaluation using in-domain open-ended questions and Claude Sonnet 4 as a judge (using the evaluation prompts from Chen et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib4 "Persona vectors: monitoring and controlling character traits in language models"))).

## Appendix H Refusal

### H.1 Data Examples

Table 5: Examples of the safety evaluation datasets

### H.2 Additional Results

![Image 40: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/gsm8k-refusal/gsm8k-vs-safety-rate-pv-layers_GSM-Danger_error-bars_annotations.png)

Figure 28: Evilness negative steering of Llama-2-7b-chat for different layers. Layer 17 is selected as the most effective.

![Image 41: Refer to caption](https://arxiv.org/html/2511.05408v2/x35.png)

Figure 29: Llama-2-7b-chat fine-tuned on GSM8K decreases the refusal to harmful queries. Safety rate (unsafety score $\leq$ 2) on DirectHarm4. The two most effective methods to restore refusals are: Weight steering with refusal data, and additional refusal examples in the training data (Joint).

![Image 42: Refer to caption](https://arxiv.org/html/2511.05408v2/x36.png)

Figure 30: Llama-2-7b-chat fine-tuned on GSM8K decreases the refusal to harmful queries. Safety rate (unsafety score $\leq$ 2) on GSMDanger. The two most effective methods to restore refusals are: Weight steering with refusal data, and additional refusal examples in the training data (Joint).

![Image 43: Refer to caption](https://arxiv.org/html/2511.05408v2/x37.png)

Figure 31: Llama-2-7b-chat fine-tuned on GSM8K decreases the refusal to harmful queries. Attack success rate (unsafety score = 5) on DirectHarm4. The two most effective methods to restore refusals are: Weight steering with refusal data, and additional refusal examples in the training data (Joint).

![Image 44: Refer to caption](https://arxiv.org/html/2511.05408v2/x38.png)

Figure 32: Llama-2-7b-chat fine-tuned on GSM8K decreases the refusal to harmful queries. Attack success rate (unsafety score = 5) on GSMDanger. The two most effective methods to restore refusals are: Weight steering with refusal data, and additional refusal examples in the training data (Joint).

### H.3 Generation Samples

#### H.3.1 DirectHarm4

#### H.3.2 GSM-Danger

## Appendix I Weight Vectors Monitoring

### I.1 Personality Datasets

We use the following prompt, system prompt and variables to generate data describing a behavior.

### I.2 Misaligned Fine-tuned Models

Table 6: Misalignment evaluation questions. We generate 20 samples for each question.

![Image 45: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/monitoring/model_comparison_dual_axis_clean.png)

Figure 33: Misalignment evaluation. Qwen2.5-7B-Instruct fine-tuned on bad advice datasets in domains medical, financial and sports. The evaluation is performed using Claude Sonnet 4 with the judge prompts from Betley et al. ([2025](https://arxiv.org/html/2511.05408v2#bib.bib41 "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs")), on the model answers to the prompts in Table [6](https://arxiv.org/html/2511.05408v2#A9.T6 "Table 6 ‣ I.2 Misaligned Fine-tuned Models ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic") (20 responses sampled for each question). All models become generally misaligned and retain generation coherence.

### I.3 Additional Results

![Image 46: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/monitoring/monitoring_w_vs_t_all.png)

Figure 34: Cosine similarity between weight vectors and task vectors.

![Image 47: Refer to caption](https://arxiv.org/html/2511.05408v2/iclr2026/images/monitoring/monitoring_tv_vs_tv.png)

Figure 35: Cosine similarity between weight vectors and task vectors.

![Image 48: Refer to caption](https://arxiv.org/html/2511.05408v2/x39.png)

Figure 36: Cosine similarity of behavior weight vectors to model weights throughout training, with emergent misalignment scores (gray, dashed). (Top) Qwen2.5-7B-Instruct fine-tuned on sports bad advice. (Bottom) Qwen2.5-7B-Instruct fine-tuned on Alpaca.

![Image 49: Refer to caption](https://arxiv.org/html/2511.05408v2/x40.png)

Figure 37: Training and validation loss curves for Qwen2.5-7B-Instruct fine-tuned on the sports bad advice dataset. Training loss is recorded at each step, while validation loss is evaluated every 5 steps. By step $\approx$ 95, where emergent misalignment appears in Figure [36](https://arxiv.org/html/2511.05408v2#A9.F36 "Figure 36 ‣ I.3 Additional Results ‣ Appendix I Weight Vectors Monitoring ‣ Steering Language Models with Weight Arithmetic"), the loss has nearly converged.
