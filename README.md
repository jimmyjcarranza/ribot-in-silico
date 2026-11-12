# ribot-in-silico

**Does a language model forget its newest memories first?**

In Alzheimer's disease, recent memories are usually lost before old ones. This pattern is known as Ribot's law. This project asks whether something similar happens in a language model that is damaged little by little: do the facts it learned early in training survive more damage than the facts it learned late?

This is not medical research. The disease is used as a lens to study how a neural network stores what it knows.

> **Status: work in progress.** No results yet. This README will be updated as each stage is completed. Every number reported here will come from a notebook in this repository.

## Method

The experiment uses [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), a 12-layer language model released by EleutherAI together with 154 snapshots saved during its training. Those snapshots make it possible to tell when the model learned each fact.

1. **The exam.** A list of simple facts from the CounterFact dataset, such as a prompt and its one-word answer. A fact counts as known when the first token of the correct answer is the model's top prediction. Only facts the fully trained model gets right stay in the study.
2. **The age of each memory.** The same exam is given to a series of training snapshots. The age of a fact is the first snapshot from which the model gets it right and keeps getting it right.
3. **The damage.** Random neurons of the final model are switched off in growing, nested levels, repeated with several random seeds. The resistance of a fact is the damage level at which the model stops getting it right.
4. **The verdict.** Age is compared against resistance: a rank correlation, and survival curves by cohort (facts learned early, in the middle, and late).

### The confound and the control

Facts learned early also tend to be the ones repeated most often in the training text, and a heavily repeated fact may resist damage for that reason alone. The confidence of the healthy model (the probability it gives to the correct answer) is used as a proxy for this. Early and late facts are compared within groups that the healthy model answered with the same confidence. If the difference disappears, the result is that resistance is explained by confidence and not by age, and it is reported as such.

## Results

Not available yet.

## Limitations

To be written with the results. Known from the design: one model family, one model size, one type of damage (random neuron ablation), one fact dataset, and an "age" measure that depends on which snapshots are sampled.

## Repository

| Path | Contents |
|---|---|
| `notebooks/01_hello_pythia.ipynb` | Smoke test: load the model, score one fact, load earlier snapshots |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyjcarranza/ribot-in-silico/blob/main/notebooks/01_hello_pythia.ipynb)

## Related work

- Biderman et al. (2023), [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf). The models and training snapshots used here.
- Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). The CounterFact dataset.
- Li, Sheng, Cohen and Pakhomov (2024), [Too Big to Fail: Larger Language Models are Disproportionately Resilient to Induction of Dementia-Related Linguistic Anomalies](https://aclanthology.org/2024.findings-acl.380).
- Alexos et al. (2024), [Neural Erosion: Emulating Controlled Neurodegeneration and Aging in AI Systems](https://arxiv.org/abs/2403.10596).
- Ellis and Lambon Ralph (2000), Age of acquisition effects in adult lexical processing reflect loss of plasticity in maturing systems: insights from connectionist networks. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 26(5), 1103-1123.
- Sohrabi (2019), [Age of Acquisition Effect: Evidence From Single-Word Reading and Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484193/). Lesions a small word-reading network and reports fewer errors on early-learned words.
- Chang, Tu and Bergen (2024), [Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability](https://arxiv.org/abs/2308.15419).

This section is a starting list, not a complete review.

## Data and licenses

- Pythia models: Apache 2.0, EleutherAI.
- CounterFact: from the [ROME project](https://github.com/kmeng01/rome) (MIT license). The dataset is downloaded at run time from its original source and is not redistributed in this repository.
- Code in this repository: MIT.

## AI assistance

This project was built with the help of Claude (Anthropic), used for experiment design discussion, code drafting and review. All results come from notebooks run by the author.

## Author

Jimmy Carranza
