# Reference

Details behind the [README](README.md). Run every command from the repository
root.

## Environments

| Environment | Used for | Spec |
|---|---|---|
| decoding (venv, Python 3.11; torch 2.5.1, transformers 4.41.2) | decoding, search baseline, corpus building | `envs/requirements.txt` |
| + JAX 0.10.2 (same venv) | HMM training | `envs/requirements-train.txt` (pip reports a cuDNN conflict; expected) |
| vLLM (venv, Python 3.11; vLLM 0.11, transformers 4.57) | sampling training data; XGrammar baseline; every gpt-oss-20b step except HMM training | `envs/vllm-requirements.txt` |
| `eai-eval` (conda) | the EAI evaluator | `envs/eai-eval.yml` |

```bash
python3.11 -m venv .venv-vllm && .venv-vllm/bin/pip install -r envs/vllm-requirements.txt
```

**gpt-oss-20b** decodes with its MXFP4 weights (about 14 GB, through the
`kernels` package and triton 3.4) and needs transformers 4.57, so its runs use
the vLLM environment as `PY`. Building its training corpus embeds the prompts
in bf16 (about 42 GB), split over the visible GPUs.

**Evaluator.** `scripts/setup_evaluator.sh` installs `eai-eval` 1.0.5
(`embodied-agent-interface` commit `531c62f`) and iGibson from the EAI fork
(commit `a4f6021c`, with two build fixes in `envs/`), downloads the iGibson
scenes (about 30 GB) and prints the python to pass as `EVAL_PY`. It needs
conda, cmake, a C++ compiler and libegl1. BEHAVIOR renders with EGL on the
job's GPU; `GIBSON_DEVICE_ID` picks another.

**Classical planner.** `pip install up-fast-downward`, or set
`FAST_DOWNWARD` to a `fast-downward.py`.

**Hardware.** One GPU with 40 GB per job; large BEHAVIOR lookahead tables add
several GB. If `/tmp` is a small RAM disk, set `TMPDIR` (JAX caches kernels
there). Hours per run on one L40S, generation plus evaluation:

| | trained | uniform | mask | syntax | trained, gpt-oss |
|---|---|---|---|---|---|
| VH AS | 2.5 | 2.5 | 3 | 0.75 | 3.5 |
| VH SD | 4 | 4 | 4 | 0.2 | 5 |
| BEH AS | 3 | 3 | 3 | 0.5 | 4 |
| BEH SD | 8 | 7.5 | 7.5 | 0.2 | 7.5 |

Runs are independent (one per GPU) and resume from the tasks already written.
`scripts/run_all.sh` runs all the Llama runs.

## Layout

```
generate_{vh,beh}_{as,sd}.py   one driver per benchmark and module
evaluate.py                    runs the EAI evaluator on a generation file
ctrlg/gamma/                   syntax automata over LM tokens
ctrlg/beta/                    semantic automata over actions; the lookahead DP
ctrlg/two_level/               the logits processor that combines them; parsers
ctrlg/schemas/                 action sets of each benchmark
ctrlg/prompts.py               the chat prompt (decoding and HMM samples share it)
ctrlg/lm_family.py             base-LM specifics: gpt-oss loading, prompt, reasoning channel
ctrlg/models.py, runner.py     HMM loading; command line shared by the drivers
baselines/                     search without an LM, Fast Downward, XGrammar
training/                      sampling, corpus building, HMM training
scripts/                       run.sh, run_all.sh, calibrate.sh, tables.py, download.sh,
                               setup_evaluator.sh, evaluate_run.sh
tests/check_guarantee.py       replays plans through their automata
data/                          prompts, vocabularies, splits, BEHAVIOR taxonomy
envs/                          environment specs
```

## Data

Ships with the repository.

| Path | Contents |
|---|---|
| `data/prompts/{virtualHome,behavior}/` | EAI prompts: 342 VH AS, 338 VH SD, 100 BEH AS, 100 BEH SD tasks |
| `data/prompts/system_prompts.json` | each module's system prompt |
| `data/prompts/augment/` | augmented prompts the HMM training samples are drawn from |
| `data/splits/` | identifiers sampled for HMM training (`train_identifiers`) and held out (`test_identifiers`) |
| `data/vocab/` | HMM vocabularies: 132 symbols for AS, 123 for SD |
| `data/taxonomy/` | BEHAVIOR object taxonomy (affordances) |

## Models and artifacts

`scripts/download.sh <name>` fetches from the Hugging Face dataset
`gwenweng/meta-ctrl-eai`, except `lm`. `LM=gpt-oss` selects gpt-oss-20b for
`lm` and `samples`.

| Name | Contents | Size |
|---|---|---|
| `lm` | Llama 3 8B Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`, gated) or gpt-oss-20b (`openai/gpt-oss-20b`) | 16 GB / 14 GB |
| `hmms` | trained HMMs for AS and SD (H = 128) for each LM, and uniform HMMs | 7 MB |
| `outputs` | our generations (`outputs/`) and evaluator outputs (`results/`) | 1.6 MB |
| `samples` | the LM's samples the HMMs were trained on (gpt-oss: final channel only) | 0.12 GB / 2.3 GB |
| `corpora` | the Llama HMM training corpora | 3.4 GB |

A trained HMM is `hmm_params.pt` (transitions, emissions) plus
`nn_checkpoint.pt`, a linear map from the LM's last hidden state at the end of
the prompt to the initial state.

**Released runs.** Named `<lm>_<module>_<arm>` (`llama`: all four arms;
`gpt-oss`: `trained`), plus `search_*`, `planner_*`, `xgrammar_*`,
`calib_llama_c{3,4}` and `calib_gpt-oss_c{2,3,4}`; `scripts/tables.py` prints
them as the paper's tables (Meta-Ctrl rows of Tables 1, 2, 6 and 8, Appendix
I). The paper's other rows (other models, the leaderboard) are not here.

**Output format.** VirtualHome runs are one JSON list; BEHAVIOR runs are a
folder of `prompt_<i>.json`, with `merged.json` (the evaluator's input) and
`batch_summary.json` (decoding statistics). Each record has `identifier` and
`llm_output`, and says how the task was decoded:

| Field | Meaning |
|---|---|
| `mode` | `two_level`, or `syntax_mask` (syntax automaton only) |
| `fallback` | why a two-level arm used syntax only (DESIGN.md, "Fallbacks") |
| `goals_dropped` | goals dropped to fit the automaton's size limit |
| `dp_budget` | plan length the lookahead covered |

## Running

`scripts/run.sh <module> <arm>` (variables `PY`, `EVAL_PY`, `N`, `LM`) writes
`outputs/<LM>_<module>_<arm>` and `results/<LM>_<module>_<arm>[_s<k>]_eval`.

- Arms: `trained` (trained HMM), `uniform` (counts valid plans), `mask` (only
  checks a valid plan exists), `syntax` (syntax automaton only).
- With `N`, use at least 51 on VH AS: its evaluator needs a task of each goal
  type, and the prompts are ordered by goal type.
- VirtualHome is evaluated six times (PYTHONHASHSEED 0–5) and averaged
  (DESIGN.md, "Evaluation notes"). Decoding is greedy.
- BEH SD `trained` uses `--ctrlg_scale 2` for both LMs, chosen by
  `LM=<lm> scripts/calibrate.sh` (c = 2, 3, 4 on 20 tasks).
- gpt-oss writes a reasoning channel before its answer; it is unconstrained and
  not scored, and the constraints act from the answer on (DESIGN.md).

**Guarantee check** (optional; scoring uses only the EAI evaluator).
`python tests/check_guarantee.py --module <module> --outputs <output> --hmm <HMM dir>`
(`--hard_mask` for `mask`; `--base_model` and `--ctrlg_scale` as decoded) reports actions the semantic automaton rejects and
plans it does not accept; both are zero on every released two-level run.

**Baselines.** Defaults match the paper; `--help` lists options.

```bash
python baselines/search_no_lm.py --module vh_as --output_dir outputs/search_vh_as
python baselines/pddl_planner_vh_as.py        # EAI_ROOT = embodied-agent-interface checkout
python baselines/xgrammar_vh_as.py            # vLLM environment
```

To evaluate one, place it where `scripts/evaluate_run.sh <name> <module>`
looks: `outputs/<name>.json` (VirtualHome) or `outputs/<name>/merged.json`
(BEHAVIOR).

## Training the HMMs

```bash
SAMPLE_PY=<vLLM python> PY=<decoding python> TRAIN_PY=<decoding python with JAX> \
  training/train_meta_hmm.sh as            # or sd; LM=gpt-oss with PY=<vLLM python>
```

1. `training/sample.py`: LM answers to the augmented prompts (T = 1). Llama:
   1M per benchmark, 200 tokens. gpt-oss: 200k per benchmark, up to 1,536
   tokens at reasoning effort low, then `training/final_channel.py` keeps the
   answer.
2. `training/build_dataset.py`: keeps answers that parse, converts them to
   action sequences, embeds the prompts.
3. `training/balance.py`: equal size per simulator.
4. `training/train.py`: H = 128, AdamW, at most 20 epochs, early stopping.

The result lands in `models/meta-hmm-<as|sd>[-gpt-oss]-retrained/`; pass it with
`--semantic_hmm_path`. To skip sampling, `scripts/download.sh samples` and set
`SAMPLES=work/samples`. To train on the released corpora
(`scripts/download.sh corpora`):

```bash
python training/train.py --data_dir work/corpora/as/behavior_AS_beta \
  --data_dir work/corpora/as/virtualhome_AS_beta_sub151682 \
  --vocab_json data/vocab/hmm_vocab_as_actions_only_full_schema.json --save_dir models/meta-hmm-as-retrained \
  --hidden_size 128 --hmm_lr 0.1 --nn_lr 0.001 --batch_size 256 --epochs 20 --patience 5 --seed 42
```

For SD: `work/corpora/sd/behavior_SD_beta`,
`work/corpora/sd/virtualhome_SD_beta_sub229967`,
`data/vocab/hmm_vocab_sd_predicates_union.json`. `--deterministic` makes GPU
training reproducible on one GPU type (about 2.5x slower).
