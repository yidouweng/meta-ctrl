# Meta-Ctrl

Code for *Meta-Ctrl: Guaranteed Plan Generation by Decoupling Syntactic and
Semantic Constraints* (CoRL 2026).

Meta-Ctrl is constrained decoding for LLM planners. A syntax automaton keeps
the LM's output well formed; a semantic automaton over actions encodes the
task's preconditions and goals; a small HMM over actions looks ahead, so the
LM is steered toward plans that are valid, reach the goal, and stay likely
under the LM.

This repository has the decoder and our experiments on the Embodied Agent
Interface (EAI) benchmark with two base LMs, Llama 3 8B Instruct and
gpt-oss-20b: VirtualHome and BEHAVIOR, action sequencing (AS) and subgoal
decomposition (SD).

## Quick start

Needs one GPU with 40 GB and Python 3.11.

```bash
# 1. Environments: the decoder, and the EAI evaluator (conda; prints its python)
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
IGIBSON_ACCEPT_TERMS=1 scripts/setup_evaluator.sh

# 2. The LM and the trained HMMs (Llama 3 is gated: accept its license and `huggingface-cli login`)
scripts/download.sh lm
scripts/download.sh hmms

# 3. Generate plans and evaluate them
EVAL_PY=<evaluator python from step 1> scripts/run.sh vh_as trained
```

`run.sh <module> <arm>` takes a module (`vh_as`, `vh_sd`, `beh_as`, `beh_sd`)
and an arm (`trained` is Meta-Ctrl; `uniform`, `mask` and `syntax` are the
ablations). Plans go to `outputs/`, scores to `results/`. Add `N=51` for a
quick run on the first 51 tasks.

For gpt-oss-20b, set `LM=gpt-oss` for both `scripts/download.sh lm` and
`scripts/run.sh`, and run with the vLLM environment
([REFERENCE.md](REFERENCE.md#environments)) as `PY`.

To look at our outputs instead: `scripts/download.sh outputs`, then
`python scripts/tables.py`.

## Where to look

| To work on... | Look at |
|---|---|
| the task rules (preconditions, effects, goals) | `ctrlg/beta/`: `vh_state_machine.py` (VH AS), `sd_state_machine.py` (VH SD), `beh_state_machine.py` (BEH AS), `beh_sd_state_machine.py` (BEH SD); [DESIGN.md](DESIGN.md) |
| the output format | `ctrlg/gamma/topdown/` (syntax automata) and the parsers in `ctrlg/two_level/` |
| the LM | `--base_model`; `ctrlg/lm_family.py` (loading, chat format, end token; gpt-oss's reasoning channel) and `ctrlg/prompts.py`. The `uniform` arm works as is; `trained` needs an HMM trained on the new LM |
| the HMM | `training/train_meta_hmm.sh` samples the LM, builds the corpus and trains |
| the benchmark | copy `generate_vh_as.py`; DESIGN.md, "Building a new task" |
| the lookahead | `--ctrlg_scale` (strength), `--max_budget` (plan length) on any `generate_*.py` |
| the baselines | `baselines/`: search without an LM, a classical planner, XGrammar |

## More

- [DESIGN.md](DESIGN.md): how the two levels fit together, writing semantic
  automata, when decoding falls back to syntax only.
- [REFERENCE.md](REFERENCE.md): environments, run times, repository layout,
  data, models, released artifacts, output format, training, baselines.

## Citation

```bibtex
@inproceedings{yidouweng2026metactrl,
  title     = {Meta-Ctrl: Guaranteed Plan Generation by Decoupling Syntactic and Semantic Constraints},
  author    = {Yidou-Weng, Gwen and Sun, Edward and Ma, Tianyi and Dogan, Metin Alp and
               Wang, Benjie and Peng, Allen and Van den Broeck, Guy and Cui, Yuchen},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

See [LICENSE](LICENSE).
