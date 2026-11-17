# Design

## The two levels

A plan is a sequence of grounded actions, written by the LM as tokens.

- **Syntax automaton γ** (over tokens) accepts exactly the well-formed outputs
  over the prompt's actions and objects.
- **Semantic automaton β** (over actions) tracks the parts of the world state
  the task's preconditions and goals depend on. It allows an action only if
  its preconditions hold, and accepts a plan only if the goals hold at the end.
- **Lookahead.** An HMM over actions approximates the LM. A backward pass over
  (β state, HMM state, steps left) gives, for each candidate next action, the
  probability of finishing with a valid plan. The decoder multiplies the LM's
  next-token distribution by it and decodes greedily.

The arms differ only in the lookahead's weights: `trained` uses the trained
HMM, `uniform` counts valid plans, `mask` only checks one exists, `syntax`
drops β.

**Reasoning models.** gpt-oss-20b writes an analysis channel, then
`<|channel|>final<|message|>` and its answer. The analysis is generated
unconstrained (reasoning effort low; after 1,024 tokens the final channel is
opened) and is not scored; both automata start at the final channel
(`ctrlg/lm_family.py`). Its HMM is trained on the final channel of its own
samples.

## Writing semantic automata

Most task knowledge lives in `ctrlg/beta/`.

- **Sound first.** Every action β allows must be executable, and every plan it
  accepts must meet the goals; the guarantee rests on this. β need not be
  complete: a rule stricter than the environment only removes some valid
  plans. When unsure, be strict.
- **Built from the prompt alone:** its objects, initial state, goals and
  action definitions. No rule reads a gold plan or a task identifier. The
  VH AS automaton (`vh_state_machine.py`) keeps preconditions and effects in
  tables, the easiest form to read and extend.
- **Check against the simulator.** Replay sampled or search-generated plans
  and look for actions β allows but the simulator rejects (a bug) and plans
  the simulator accepts but β rejects (a gap). Develop rules on training
  prompts (`data/splits/`).
- **Track only what matters.** β's size grows with each fact it tracks, so it
  tracks only objects and properties a goal or precondition refers to.
- **Limits.** Each module caps β's size; past the cap, goals are dropped until
  it fits (recorded as `goals_dropped`). The lookahead covers a fixed plan
  length: 20 actions (VH), 30 (BEH AS), 35 (BEH SD).

Deliberately outside β: seven BEHAVIOR AS actions the evaluator does not
execute reliably (`BLOCKED_ACTIONS` in `generate_beh_as.py`), BEHAVIOR goal
forms not in `SIMPLE_GOALS` (such as nested quantifiers), and a few VH AS
actions no goal needs (`ctrlg/schemas/vh_as.py`).

### Fallbacks

If β exceeds its cap, or no accepting plan fits in the plan length, the task
is decoded with γ alone (`mode: syntax_mask`): well formed, but semantics not
enforced. If decoding reaches a dead end, the decoder still masks every action
β forbids. The guarantee holds for every task not decoded with γ alone. In
this release, 22 of 880 tasks fall back, the same in every arm and for both LMs:

| Reason | Tasks |
|---|---|
| β exceeds its size cap | 6 |
| no accepting plan within the plan length | 5 |
| no output the prompt allows meets the goal | 4 |
| β stricter than the simulator | 7 |

## Building a new task

`generate_vh_as.py` is the template.

- **γ.** Subclass `ctrlg.gamma.topdown.base.CanonicalStateMachine` and compile
  it with `ctrlg.gamma.topdown.compile.compile_topdown`.
- **β.** An object with `transitions`, `accept_mask`,
  `get_initial_state_id()`, `is_accept(state)` and `apply(state, step)` (next
  state, or `None` if not allowed). `ctrlg.beta.backward_dp.BetaBackwardDP`
  builds the lookahead from it.
- **Parser and candidates** (`ctrlg/two_level/`): turn a token block into an
  action, and list the actions a partial block can still become.
- **Vocabulary.** `ctrlg.beta.backward_dp.SemanticVocab` maps an action to HMM
  symbols. A uniform HMM needs no training; for a trained one, sample your LM
  on training prompts and use `training/`.

While developing, `tests/check_guarantee.py` (optional, not part of scoring) replays outputs through the
automata and reports any action β rejects or plan it does not accept, which would point to a decoder bug.

## The HMM

One HMM per module type (AS, SD), trained on both simulators. Its initial
state comes from a linear map of the LM's last hidden state at the end of the
prompt; in practice this separates the two simulators. On BEH SD the
lookahead is raised to the power c = 2 (`--ctrlg_scale`), chosen on 20 tasks;
elsewhere c = 1.

## Evaluation notes

The EAI VH AS evaluator matches action goals in Python set order, so its
verdict depends on the string-hash seed. Every VirtualHome output is
evaluated with PYTHONHASHSEED 0–5 and averaged. Decoding pins
PYTHONHASHSEED = 0.
