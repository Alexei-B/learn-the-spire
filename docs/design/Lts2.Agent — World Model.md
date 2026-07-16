# Lts2.Agent — World-Model Agent Design (encoder / predictor / planner)

Status: **design only — nothing implemented.** This doc turns the "encoder/decoder + predictor +
ranker" idea into a concrete, phased architecture, grounded in the model-based-RL literature and in
what we measured from the current PPO pipeline. It is the input to future implementation plans; it
deliberately does not specify code layout.

Companion docs: `Lts2.Agent — Protocol.md` (the wire), `Lts2.Harness.md` (the emulator).
Current baseline: `python/lts2_agent/` (PyTorch PPO, per-option actor-critic, hand-built features).

---

## 1. Problem statement

Two coupled problems with the current agent:

1. **Sample inefficiency.** A representative run: ~1,840 iterations × ~1,536 steps ≈ **2.8M combat
   decisions (~100k fights)** with win-rate flat around 0.6 and hpLost ~35 on the scenario task.
   PPO is on-policy: every transition is used for a handful of gradient steps and thrown away, and
   the *only* training signal is the scalar reward. Nothing in the loss ever asks the network to
   understand what a card *does* — it must infer "Unleash scales with Osty's HP, therefore summons
   are also damage" from reward correlations across thousands of noisy fights.
2. **Wall-clock per sample.** The emulator is fast for what it is (~15–25 ms/step, ~600 steps/s
   across 16 host processes), but model-free RL needs so many samples that this is still hours per
   experiment. The GPU sits idle ~95% of the time; the bottleneck is C# env stepping and JSON I/O.

There is also a third, structural problem the user called out directly:

3. **The feature treadmill.** `features.py` hand-encodes ~19 card scalars + hashed power buckets.
   Every mechanic the encoding doesn't surface (Osty-scaling, orbs, enchantments, star costs,
   power interactions) is invisible until someone adds a feature for it, bumps `FEATURE_VERSION`,
   and retrains. This does not scale to the full card pool and it caps what the policy can learn.

The proposal attacks all three: (3) with a learned encoder over the *complete* state, (1) with a
self-supervised predictor that extracts training signal from every transition regardless of reward,
and (2) by moving most computation from the emulator to GPU-batched latent rollouts.

## 2. The proposal, mapped to prior art

The three-part sketch is a **world-model agent**, and each part has a name and a decade of results
behind it. That's good news: the idea is sound, and the known failure modes are documented.

| Proposal component | Literature name | Closest systems |
|---|---|---|
| Encoder/decoder over full game state | Representation learning for world models | World Models (Ha & Schmidhuber 2018), DreamerV3's RSSM encoder/decoder |
| Predictor: per-action next-latent + reward + legal actions | Learned **dynamics function** | MuZero (Schrittwieser et al. 2020); latent-only prediction is JEPA (LeCun 2022) |
| Predictor for "end turn" / random effects | **Afterstates + chance nodes** | **Stochastic MuZero** (Antonoglou et al., ICLR 2022) |
| Ranker over predicted states | **Value function** (+ policy prior) | TD-Gammon afterstate values (Tesauro 1995); MuZero's value/policy heads |
| Chained predictions, à la AlphaZero tree search | **Planning in the learned model** | MuZero MCTS; DreamerV3 "imagination" rollouts |
| "Predictor learns from any input/output pair" | Self-supervised model learning from a **replay buffer** (off-policy) | All of the above; MuZero Reanalyse |

Three load-bearing findings from that literature shape the design below:

- **MuZero deliberately has no decoder.** Its latent is trained *only* to predict reward, value,
  and policy along real trajectories ("value-equivalent" models) — reconstruction proved
  unnecessary for Atari/Go and can even waste capacity on decision-irrelevant detail. **We should
  keep the decoder anyway**, for reasons specific to us (§4.3): our observations are small
  structured records, not pixels, so decoding is cheap and exact; the decoded prediction is a
  one-of-a-kind debugging tool; and the decoder is what lets the predictor answer "what actions
  will be available next" (the user's requirement) almost for free. But the MuZero lesson stands:
  **reconstruction alone is not enough** — the latent must *also* be trained on reward/value
  prediction, or the encoder will happily preserve information the planner doesn't need while
  blurring the details it does.
- **JEPA-style decoder-free latent prediction has a collapse failure mode** (encoder and predictor
  agree to map everything to a constant). Symbolic reconstruction is our anti-collapse anchor —
  another reason to keep the decoder.
- **Stochastic MuZero's afterstate factorization is exactly our End-Turn problem.** It splits every
  transition into a deterministic *afterstate* step (the consequence of the action) and a *chance*
  step (a distribution over discrete outcomes), and runs MCTS with chance nodes. Backgammon and
  2048 — its benchmark domains — have the same shape as an STS turn: your move is deterministic,
  then dice/spawn randomness intervenes. This resolves the user's uncertainty question in a
  principled way (§4.6).

**The AlphaZero comparison, made precise:** AlphaZero searches with the *real* rules; MuZero is
AlphaZero with the rules replaced by a learned latent predictor — which is exactly the proposal
here. The user's instinct that "if you can plan on predictions you can attack far messier games"
is the actual thesis of the MuZero → Stochastic MuZero → DreamerV3 line of work.

## 3. What kind of problem is STS2 combat, exactly?

Design follows from domain structure, so pin it down (all verifiable against `GameState.cs` and
the wire protocol):

- **Near-perfect information.** The state DTO exposes players, hand, draw/discard/exhaust piles,
  powers, orbs, Osty, relics, potions, enemies with HP/block/powers and — crucially —
  **telegraphed intents** (damage × hits). The only hidden information is *ordering/roll* RNG:
  the shuffle order of the draw pile (contents are known as a multiset), unrevealed randomness
  (Discovery's three offered cards), and some enemy AI rolls. This is far more observable than
  backgammon; a learned model has almost everything it needs in-frame.
- **Stochasticity is localized.** Within a turn, most card plays are exactly deterministic given
  the visible state (damage/block previews are already power-adjusted server-side). The chance
  points are: **End Turn** (enemy action resolution + next-turn draw + new intents), draw/shuffle
  effects mid-turn, and "offer" effects (Discovery, some events). Everything else is an
  afterstate.
- **Actions are a variable, structured set:** (card instance × target) + potions + EndTurn +
  mid-effect `SelectCards` choices. The user's insight that sub-choices (Discovery's 3 options, a
  forced enemy choice) are *just states with different option sets* is correct and already how the
  harness models them (`GamePhase.Choice` → `SelectCards` options). No special-casing needed:
  the predictor treats a choice-state like any other state.
- **Episodes are short and reward is simple.** A fight is ~25–30 decisions; the objective the user
  wants is "end the fight having lost the least HP, don't lose." That admits a clean, bounded,
  interpretable value target (§4.5).
- **We own a perfect simulator — with two constraints.** (a) **No mid-combat snapshot/restore**
  (combat state lives in `CombatManager`; `GameHost.Snapshot()` is out-of-combat only), so
  simulator-based tree search must *replay* the fight from the start for every branch. (b)
  **Replaying a seed reveals the true RNG** — a simulator search would "see" the actual shuffle
  order and Discovery offers, i.e. it plans with information the agent shouldn't have, unless we
  re-randomize hidden state per branch (determinization), which the harness doesn't support today.
  These two constraints are why a *learned* predictor is not redundant with the simulator (§6.B).

## 4. Recommended architecture

Five modules. Names chosen to match the proposal: **Tokenizer → Encoder → Decoder → Predictor →
Value/Policy (the "ranker") → Planner.**

```
                         ┌────────────┐  reconstruct s (all entities, fields)
            s (JSON) ──► │  Encoder   │──► z ──► Decoder ─► ŝ            [trains encoder; debug]
                         └────────────┘   │
                                          │        a = (card, target) token
                                          ▼
                                   ┌─────────────┐
                                   │  Predictor  │  deterministic step:  z, a ─► z_after, r̂, t̂
                                   │ (dynamics)  │  chance step:         z_after ─► P(c), then
                                   └─────────────┘                       z_after, c ─► z'
                                          │
                                          ▼
                              Decoder(z') ─► ŝ'  (next state incl. hand ⇒ next legal actions)
                                          │
                                          ▼
                         ┌──────────────────────────────┐
                         │  Value V(z) + Policy π(a|z)  │   the "ranker", refined
                         └──────────────────────────────┘
                                          │
                                          ▼
                         Planner: greedy → intra-turn search → MCTS w/ chance nodes
```

### 4.1 Tokenizer: the state as a set of entity tokens (kills the feature treadmill)

Replace the fixed scalar vectors with a **set of typed tokens**, one per game entity:

- **Card tokens** — one per card in hand, and (as unordered multisets) draw pile, discard, exhaust.
  Content: card-id embedding + the full static catalog row (tags/keywords/var-keys — already dumped
  via `--dump-cards`) + live dynamic fields (cost, damage/block/summon previews, upgraded, X-cost,
  enchant/affliction, replay count) + a "zone" embedding (hand/draw/discard/exhaust/offered).
- **Creature tokens** — player, Osty, each enemy: HP/maxHP/block, and **their powers as child
  tokens** (power-id embedding + amount — replacing the lossy 16-bucket CRC hash; dump a power
  catalog exactly like the card catalog).
- **Intent tokens** — per enemy intent: type, damage, hits, target.
- **Global token** — energy/maxEnergy, stars, turn number, orb slots, relic set (multi-hot or relic
  tokens), potion tokens, act/floor.
- **Pending-choice token(s)** — when the state is a `SelectCards` choice: what's being offered,
  how many to pick, the source effect.

Rule: **if the wire exposes it, tokenize it.** No more per-mechanic feature engineering — new
mechanics arrive as new ids in the catalogs plus generic numeric fields. The draw pile must be
encoded as an *unordered* multiset even though the wire may order it — the agent must not be able
to read the shuffle.

This tokenizer is useful *regardless of everything else in this doc*: even the current PPO model
would improve with attention over tokens instead of mean-pooled scalars. It is Phase 1 for that
reason (§9).

### 4.2 Encoder

A small **set transformer** (permutation-invariant attention; ~2–6 layers, d≈128–256, a few M
params — the current 128-wide MLP is tiny by comparison and still leaves the GPU idle) over the
token set, producing:

- per-token output embeddings (used by the decoder and by per-action scoring), and
- a pooled **latent state `z`** (the CLS-style summary the predictor and value head consume).

Attention is what solves the Unleash example *structurally*: the Unleash token (whose catalog row
carries the `OstyDamage` var-key) can attend to the Osty token's HP; "Bodyguard > Defend when
Unleash is in the deck" becomes a two-hop attention pattern instead of a reward-correlation ghost.

### 4.3 Decoder — symbolic reconstruction, and why it earns its keep

The decoder reconstructs the **structured state, not pixels**: for each token slot, its
categorical ids (card id, power id, intent type) and numeric fields (HP, block, energy, amounts),
plus set sizes. Losses are cross-entropy on ids, regression (or two-hot, DreamerV3-style) on
numerics. This is cheap and *exactly* measurable — per-field accuracy, per-entity error.

Three jobs, in priority order:

1. **Grounding / anti-collapse:** guarantees `z` retains the full state (the JEPA failure mode is
   impossible when you must reproduce the enemy's HP from `z`).
2. **Legal-action prediction for free:** in STS2, the legal action set is (almost) a function of
   the visible state — playable hand cards × live targets, potions, EndTurn; choice-states list
   their offered options. So decoding the predicted next state (hand, energy, enemies alive,
   pending choice) *is* the "predict available actions" requirement from the proposal. A thin
   correction head can patch the residual cases (unplayable flags, weird keywords); measured
   against real `ListOptions` output.
3. **The debugger we've never had:** decode the predictor's output and print it. "The model thinks
   playing Unleash leaves Lagavulin at 4 HP and draws no cards" is human-checkable per-field —
   a categorically better diagnostic than staring at win-rate curves (compare the pain in the
   PPO diagnostics saga, where every hypothesis needed a bespoke eval).

Caveat carried over from MuZero (§2): reconstruction is the *auxiliary*; the latent must also be
shaped by reward/value/policy losses so decision-relevant information is prioritized, not merely
present.

### 4.4 Predictor (the dynamics model) — deterministic afterstate + discrete chance step

Factor every transition, per Stochastic MuZero:

1. **Afterstate step (deterministic):** `f(z, a) → z_after, r̂, t̂` — the consequence of the action
   itself: energy paid, damage previewed, block gained, card moved to discard, enemy died, choice
   opened. `r̂` predicts the immediate reward components (ΔplayerHP, ΔenemyHP, kills — see §4.5),
   `t̂` predicts fight end. The action `a` is encoded from the *card token + target token* — the
   same embeddings the encoder produced, so action understanding is shared with state
   understanding. For most card plays this step is the whole transition and its training loss can
   go to ~zero, giving the model an easy, verifiable core of the rules.
2. **Chance step (stochastic, discrete):** `g(z_after) → P(c)` over a small learned codebook of
   **chance outcomes** `c` (size ~32–64, VQ-style as in Stochastic MuZero), then
   `h(z_after, c) → z'`. The chance code is trained to encode what *actually* happened in the
   logged transition (which cards were drawn, which intent roll occurred, which three cards
   Discovery offered) — so at training time `c` is inferred from `s'` (posterior), and at planning
   time `P(c)` is sampled/enumerated (prior). Card plays with no randomness learn a collapsed
   `P(c)` (one code, probability 1) — the model itself discovers *which* actions are stochastic.
   **End Turn is not special-cased:** it is simply an action whose afterstate step covers the
   deterministic part (telegraphed intent damage resolving into HP/block — mostly predictable!)
   and whose chance step carries the draw + next intents. We can optionally *semi-supervise* the
   End-Turn chance code with the known drawn-card ids to speed learning.

The predictor must support **K-step unrolled training** (MuZero uses K=5): encode `s_t`, unroll
through the logged actions `a_t..a_{t+K-1}`, and apply reconstruction/reward/value losses at every
step against the logged `s_{t+1}..s_{t+K}`, with the chance codes teacher-forced from the logged
outcomes. This is what keeps compounding error in check and is the single most important training
detail; one-step-only training produces models whose 3-step rollouts are garbage.

Two post-2021 refinements to bake in from the start (see §11 for provenance):

- **Latent consistency loss** (EfficientZero's key ingredient, also TD-MPC2's): at every unroll
  step, pull the *predicted* latent `ẑ_{t+k}` toward the *encoder's* latent of the real
  `s_{t+k}` (stop-gradient on the target, SimSiam/EMA style). This supervises the dynamics in
  latent space directly instead of only through the decoder, and was the single biggest
  sample-efficiency lever in the EfficientZero ablations.
- **Normalize the latent** (TD-MPC2's SimNorm, or DreamerV3's categorical latents): a bounded,
  normalized `z` cheaply prevents both latent explosion and degenerate collapse — the same class
  of runaway we already fought once in the PPO value head.

Where the model complexity budget goes: exactly as the proposal says, **most parameters belong
here** — the predictor is where the game's rules live.

### 4.5 Value + policy — the "ranker", made precise

The proposal's least-developed part has a standard answer: the ranker is a **state-value function**
`V(z)`, plus (slightly less obviously) a **policy prior** `π(a|z)`. Both are small heads on the
latent; the intelligence is upstream.

- **Value target.** Define the return of a fight as its terminal outcome only:
  `R = hp_end / hp_max` on a win, `R = −λ` on a loss/timeout (λ ≈ 0.5–1), optionally minus a tiny
  per-turn cost. Then `V(z) ∈ [−λ, 1]` is literally "expected fraction of HP I'll finish this
  fight with" — bounded (no more value-head blow-ups à la the tanh/±20 clamp saga), interpretable,
  and exactly the user's stated objective ("the best game state is mostly just losing the least
  HP"). Intermediate shaping (dense damage rewards, step penalties) has repeatedly distorted play
  (see the PPO reward-misalignment findings); with a model-based critic and short fights, we can
  afford the sparse, honest objective. Train `V` by n-step TD / λ-returns on **real** logged
  fights (later: on imagined rollouts too, §5).
- **Ranking rule (v1):** for each legal action, one predictor step, then score
  `r̂ + γ·E_c[V(z')]` — for deterministic actions that's just `V(z_after)`; for stochastic ones,
  the expectation over the chance distribution (top-k codes suffice). This is TD-Gammon's
  afterstate ranking, which was enough for superhuman backgammon *without any deep search* —
  a meaningful benchmark for how far v1 alone might go.
- **Why a policy prior too:** (a) search needs it to focus expansion (AlphaZero/MuZero's
  policy-guided MCTS); (b) it distills search results back into a fast reactive policy (train
  `π` toward the planner's improved action distribution — "expert iteration"); (c) it is the
  cheap fallback when the decision-server time budget is tight. Behavior policy during data
  collection: sample from `π` with temperature (the argmax-collapse lesson from PPO applies
  unchanged).

### 4.6 Planner — staged depth, and the answer to the Discovery problem

The user's uncertainty concern ("you can't plan through Discovery's unknown three cards") has a
crisp resolution: **plan through determinism, take expectations at chance, and let `V` absorb
everything beyond.** A chance node is not a wall — it's a weighted average over a few sampled
outcomes, each of which is again a normal state. And when even sampling is too deep/wide, the
value function *is* the summary of "how good is it to stand here, on average, over all futures."
That is exactly the division of labor in Stochastic MuZero and in DreamerV3's λ-return over
imagined stochastic rollouts.

Stage the planner; each stage is shippable and measurable:

- **P-0: Greedy afterstate ranking** (1 predictor step + value, §4.5). No search. This alone is
  structurally stronger than the current per-option logit, because the score is grounded in a
  predicted outcome rather than pattern-matched features.
- **P-1: Intra-turn search.** Within a turn, chain deterministic afterstates: beam search (beam
  ~8–16) over card sequences from the current hand — [Defend, Defend, Strike, End Turn] is a path
  — evaluating each leaf as `V(after End Turn's afterstate)` i.e. after the *predictable* part of
  the enemy turn, before the draw. Mid-turn chance actions (Discovery) become leaves scored by
  expectation, or expand top-k codes. This captures most of STS tactics (play order, energy
  budgeting, lethal detection, block-vs-damage tradeoffs) while staying inside the regime where
  the predictor is most accurate — a turn is typically ≤6 plays from a hand we can see. Note the
  legal-action prediction (§4.3.2) is what allows expansion past the first ply.
- **P-2: Multi-turn MCTS with chance nodes** (Stochastic MuZero proper), if P-1 plateaus:
  cross End Turn by sampling ~k chance codes per chance node, guided by `π`, backed by `V` —
  all GPU-batched latent ops, no emulator in the loop. Use **Gumbel search** (Gumbel MuZero's
  sequential-halving root selection, §11) rather than vanilla PUCT: it guarantees policy
  improvement even at tiny simulation budgets (~8–32 sims instead of MuZero's classic 50–800),
  which matters both for training throughput and for the decision-server latency budget when
  serving into the TUI. This is the "much more complicated games, less determinism" endgame the
  user described; it's also the point of steepest engineering cost, hence last.

The planner's output (visit counts / improved ranking) feeds back as `π`'s training target, and —
served through the existing decision-server seam — is directly watchable in the TUI.

## 5. Training regime — where the sample-efficiency and speed wins actually come from

- **A replay buffer replaces the on-policy pipeline for everything except `V`'s freshest targets.**
  Log every transition `(s, a, s', options, options')` from *any* source — random play, the
  heuristic, closed-eval scenarios, every past and present policy. The encoder/decoder/predictor
  are supervised learners: they never need on-policy data, never go stale, and extract signal from
  lost fights as efficiently as won ones ("the predictor doesn't need to win fights to get
  smarter" — correct, and it's the core economic advantage). The ~100k-fight corpus a PPO run
  burns through today would be *kept* and mined for years of gradient steps.
- **Model learning needs no reward at all**, so scenario-mode's reward-weight tuning treadmill
  (`--sw-win/--sw-loss/--sw-hp`, entropy floors, …) stops gating progress: most of the learning
  problem becomes "minimize measurable prediction error," which is the tractable, debuggable kind.
- **The emulator's new job is data collection + evaluation only.** Planning happens in latent
  space at GPU speed (a predictor step is ~µs batched, vs ~20 ms/env-step); training reads the
  buffer. Env-side crashes (the stale-card/timeout errors in the training logs) cost one lost
  trajectory, not a stalled iteration barrier.
- **Later accelerants, in order of likely value:** MuZero-Reanalyse (recompute stale value targets
  with the current net — huge for buffer reuse); DreamerV3-style imagination (train `V`/`π` on
  latent rollouts from real starting states — decouples RL from the env entirely); curiosity-style
  exploration bonuses from predictor error (optional; revisit only if data diversity stalls).
- **Curriculum still matters.** The act-0 starter-deck ceiling (win-rate capped ~0.56 regardless
  of play) hides learning in any architecture; keep scenario-mode's random decks/encounters and
  the fixed-seed greedy+sampled eval protocol exactly as-is.

## 6. Alternatives considered

**A. Stay model-free, just fix the representation** (tokenizer + transformer + an off-policy
algorithm like DQN-per-option or IMPALA). Cheaper, and honestly it would help — the representation
is plausibly the current binding constraint. **Verdict: not either/or.** The tokenizer/encoder
(Phases 1–2) *is* this alternative; if we stopped after Phase 2 with a PPO head on top, we'd still
have banked most of its value. The predictor phases are additive on top of it. Note also that the
strongest *model-free* sample-efficiency results (SPR → BBF, §11) get there by bolting a latent
**self-prediction auxiliary loss** onto the value learner — i.e. even the model-free frontier
converged on "add the predictor"; they just never plan with it. So P3's predictor pays off even in
the world where planning never ships: it doubles as SPR-style auxiliary supervision for `V`/`π`.

**B. Search with the real simulator instead of a learned predictor** (expectimax/MCTS over
`Lts2Env`). Attractive — the "model" is perfect. Three blockers, all noted in §3: no mid-combat
snapshot means every branch replays the fight from the start (O(depth) env-steps per node at
~20 ms each ⇒ seconds per root decision, ~10⁴–10⁵× slower than latent steps); same-seed replay
leaks the true shuffle/offers (planning with x-ray vision — results wouldn't transfer to honest
play) and fixing that means building determinization into the harness; and it contributes nothing
toward the "generalize to messier games" goal. **Verdict: not the agent — but a superb *oracle*.**
A slow replay-based simulator search on a few hundred fixed positions gives (a) ground-truth
next-states to score the predictor against, and (b) near-optimal action labels to benchmark the
planner's regret. Worth building as *test infrastructure* in the predictor phase.

**C. Decoder-free latent prediction (pure JEPA).** Elegant, avoids reconstruction capacity waste.
Rejected for v1: collapse risk needs careful anti-collapse machinery, and we'd forfeit the
symbolic-decode debugger and the legal-action derivation — the two most operationally valuable
by-products of the design. Revisit only if reconstruction demonstrably bottlenecks capacity.

**D. LLM-based play.** Prior art exists (LLMs play STS1 passably as text). Wrong tool here: we
want a small, fast, trainable model tightly coupled to a simulator, not frozen general priors at
~1 s/decision.

## 7. Risks and mitigations

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Compounding rollout error** — predictions degrade with depth; deep plans become fiction | Latent drift multiplies per step | K-step unrolled training (§4.4); keep planning shallow (intra-turn) until measured k-step accuracy supports more; value absorbs the horizon |
| **Planner exploits model error** — search finds actions the model *wrongly* loves (hallucinated value), play degrades as search deepens | Optimizer-vs-model adversarial dynamic; the classic model-based failure | Track "planner regret vs oracle" (§6.B) and real-env win-rate per search depth; cap depth where the curve bends; keep π/V trained on *real* outcomes as anchor |
| **Chance model misses rare outcomes** — codebook covers common draws, tail events surprise the agent | VQ codebooks under-represent tails | Semi-supervise End-Turn codes with drawn-card ids; monitor per-code perplexity + calibration of P(c); fall back to expectation-only planning at poorly-calibrated nodes |
| **Latent shortchanges decision-relevant detail** (reconstruction≠value-relevance, §4.3 caveat) | MuZero's original argument against decoders | Joint loss: reconstruction + reward + value + policy shape the same latent; weight-tune on predictor-accuracy vs win-rate evals |
| **Engineering surface area** — 5 modules, unrolled training, replay infra, per-phase evals; a lot can be silently wrong | This is the real cost vs "PPO + tricks" | Phase gates with *supervised* (hence unambiguous) exit criteria (§9); every phase independently useful; decoded-state debugger from Phase 2 onward |
| **Throughput of data collection still gates the buffer** | 600 steps/s feeds the buffer slowly at first | Buffer is append-forever (old data never stales for the model); seed it with cheap random+heuristic play at scale before any model exists |
| **Determinism/parity drift** between logged states and live serving | Same class of bug as the FEATURE_VERSION contract today | Keep the tokenizer as the single train/serve parity contract (same role `features.py` plays now), version-stamped; catalog signatures already do this for cards |

## 8. Evaluation — what "it's working" looks like per component

The decisive advantage of this design is that **most components are supervised and can be judged
without playing a single fight**:

- **Encoder/decoder:** per-field reconstruction accuracy on held-out states (card-id top-1, HP
  MAE, power amounts, option-set F1). Target: ~exact.
- **Predictor, deterministic step:** same metrics on ŝ' vs logged s', held-out transitions,
  reported *per action kind* (PlayCard vs EndTurn vs SelectCards) and at K∈{1,3,5} unroll depths.
  This directly quantifies "does the model know the rules," per rule.
- **Predictor, chance step:** log-likelihood of realized outcomes; calibration of P(c);
  draw-prediction accuracy given semi-supervision.
- **Legal-action head:** exact-set match vs `ListOptions` on held-out states.
- **Value:** calibration of V(z) vs realized final-HP-fraction, on the fixed-seed eval fights.
- **Policy/planner:** the *existing* fixed-seed greedy+sampled eval protocol (win, hpLost,
  hpFrac), unchanged, so curves are comparable with the PPO baseline; plus planner-regret vs the
  simulator oracle on a small fixed position set; plus win-rate as a function of search budget
  (the MuZero-style scaling curve — if more search doesn't help, the model is the bottleneck).

## 9. Phased plan (each phase shippable, measurable, and useful even if we stop there)

1. **P0 — Transition logging + oracle probes.** Log `(s, options, a, r-components, s', options')`
   from scenario rollouts (any policy) into a replay corpus; dump the power catalog (mirror of
   `--dump-cards`); build the replay-based simulator oracle for a fixed probe set (§6.B).
   *Exit:* ~10⁶ transitions on disk; corpus loader; probe set frozen.
2. **P1 — Tokenizer.** The full entity-token encoding (§4.1) as the new parity contract.
   *Optional quick win:* swap it under the existing PPO model (attention pool instead of
   mean-pool) to bank Alternative-A value and sanity-check the encoding end-to-end.
   *Exit:* tokens round-trip every state in the corpus; (optional) PPO-on-tokens ≥ PPO baseline.
3. **P2 — Encoder + decoder.** Set transformer + symbolic reconstruction, trained on the corpus.
   Build the decoded-state pretty-printer (the debugger).
   *Exit:* reconstruction metrics ~exact on held-out states.
4. **P3 — Predictor.** Afterstate + chance steps, K-step unrolled, reward/terminal heads,
   legal-action derivation. Score against held-out transitions and the oracle probes.
   *Exit:* deterministic-step accuracy ≳ decoder-level on 1-step, degrades gracefully to K=5;
   End-Turn chance log-likelihood beating a draw-uniform baseline; option-set F1 high.
5. **P4 — Value + greedy afterstate agent.** V trained by TD on logged fights; serve
   `argmax r̂+γE[V]` (sampled variant for collection) through the decision-server seam.
   *Exit:* beats the PPO checkpoint and the heuristic on the fixed-seed eval.
6. **P5 — Intra-turn beam search** (+ policy prior, + distillation).
   *Exit:* beats P4; regret vs oracle shrinks; win-rate-vs-beam-width curve rises.
7. **P6 — (contingent) chance-node MCTS / imagination training / Reanalyse**, only where P5's
   scaling curves say the model can support it.

Rough effort intuition: P0–P2 are bread-and-butter supervised infrastructure; P3 is the heart and
the main research risk; P4–P5 are small heads + search code on top. If P3's metrics come out weak,
we stop, keep P1–P2 (which already subsume the best model-free upgrade), and have learned exactly
*which* rules the model can't predict — per-field, per-action-kind — instead of a flat win-rate.

## 10. Open questions (deliberately deferred)

- **Latent shape:** single pooled vector vs a small set of latent tokens (per-entity latents make
  the predictor's job more local; Stochastic MuZero used flat latents; recent world models trend
  toward token sets). Decide in P2 by reconstruction quality per capacity.
- **Chance codebook size / VQ vs categorical** (DreamerV3 uses 32×32 categoricals; Stochastic
  MuZero uses VQ codes). Decide in P3.
- **How much reward shaping to keep** during V-training warm-up (sparse-only is the goal; a
  brief shaped curriculum may speed the first sessions).
- **Cross-fight scope** (Pen Nib, potions economy, map/deck decisions): explicitly out of scope,
  as the user specified — but the tokenizer should not *preclude* run-level tokens later.
- **Multi-character generality:** the corpus should mix characters from day one (scenario mode
  already does) so the predictor learns shared rules rather than one character's.

## 11. Follow-up literature scan (2022 → 2026): what the descendants of our anchors teach us

The doc's anchors are 2020–2022 papers; each spawned a line of follow-up work. Scanned July 2026.
Verdicts are relative to *this* design (small symbolic states, localized stochasticity, owned
simulator), not to the papers' own pixel/Atari settings.

### MuZero line

- **Gumbel MuZero** (Danihelka et al., ICLR 2022): replaces PUCT root selection with Gumbel top-k
  sampling + sequential halving, with a *proven* policy-improvement guarantee at any simulation
  budget — matches classic MuZero at n=800 sims and still improves at n=2–16. **Adopt outright**
  in P-2 (§4.6); it converts MCTS from "expensive endgame" to something serveable within the
  decision-server timeout. EfficientZero V2 (2024) confirms it halves search budgets in practice.
- **EfficientZero** (Ye et al., NeurIPS 2021) / **EfficientZero V2** (2024): human-level
  Atari-100k from ~2 hours of experience — the canonical proof of the sample-efficiency claim in
  §5. Its three additions, in our terms: (1) the **latent consistency loss** (now folded into
  §4.4 — their ablations show it's the biggest single lever); (2) a *value-prefix* head
  (predict cumulative reward over the unroll window rather than per-step reward — hedges timing
  ambiguity; cheap to add if per-step `r̂` proves noisy); (3) model-based off-policy value
  correction (re-bootstrap stale buffer targets with fresh model rollouts — subsumed by
  Reanalyse in our plan).
- **MuZero Unplugged / Reanalyse (2021) and ReZero (2024):** training entirely from a fixed
  offline buffer works, and reanalyze-style target refreshing is the mechanism that makes
  "append-forever replay" (§5) keep paying; ReZero is engineering to make it cheap. Validates the
  buffer-centric training economy this design bets on.
- **UniZero (2024) + LightZero (NeurIPS 2023 benchmark):** UniZero swaps the recurrent latent for
  a transformer over (state, action) token history, decoupling "latent state" from "latent
  history", and jointly trains model + policy — outperforms MuZero on 17/26 Atari-100k games.
  Relevant if we ever need cross-turn *memory* beyond what the visible state carries (mostly we
  don't: STS2's state is near-fully observable, §3). **The practical takeaway is LightZero
  itself**: an actively maintained PyTorch framework implementing MuZero, Stochastic MuZero,
  Sampled MuZero, Gumbel MuZero, and UniZero behind one interface. The implementation plan should
  seriously consider building on or at least cross-checking against it rather than
  reimplementing chance-node MCTS from scratch.

### Dreamer line

- **DreamerV3** was published in Nature (2025) essentially unchanged — its robustness recipe
  (symlog targets, two-hot regression, categorical latents, fixed hyperparameters across 150+
  tasks) is the field's consensus "make it stable without per-task tuning" toolkit; we already
  reference two-hot in §4.3 and the categorical-latent option in §10.
- **Dreamer 4** (Hafner & Yan, 2025): a 2B-param transformer world model that solves Minecraft
  "obtain diamonds" with the policy trained **entirely inside the world model from a fixed
  offline dataset** — no environment interaction during RL at all — using ~100× less labeled data
  than prior offline RL. Two transferable lessons: (1) the strongest available evidence for our
  endgame economics (env = data collection + eval only; all optimization against the buffer and
  the model); (2) its "shortcut forcing" objective works by predicting the *clean final state*
  (x-prediction) rather than deltas — our symbolic decoder predicting absolute next-state fields
  is already the x-prediction shape, and we should keep it that way (don't be tempted into
  delta/diff prediction, which compounds).
- **Token-based world models — IRIS, TWM (2023), STORM (NeurIPS 2023), Δ-IRIS (2024):**
  the field converged on exactly our §4.1–4.2 shape (tokenize → transformer → predict tokens).
  STORM is the encouraging data point on scale: a **2-layer** transformer world model matches
  DreamerV3 on Atari-100k. Our latent budget estimates in §4.2 are, if anything, generous.

### JEPA line

- **V-JEPA 2 / V-JEPA 2-AC** (Meta, 2025): action-conditioned latent prediction at scale, used
  for zero-shot robot planning — the decoder-free program does work now, in pixel domains where
  reconstruction is genuinely expensive. Our states are a few hundred symbols; reconstruction
  costs us almost nothing and buys the debugger + legal-action head, so §6.C's verdict stands.
- **TD-MPC2** (ICLR 2024): the strongest practical decoder-free recipe — latent world model
  trained on consistency + reward + value only, **SimNorm** latent normalization to prevent
  collapse/explosion, one fixed hyperparameter set across 100+ continuous-control tasks. Two
  imports even though we keep the decoder: SimNorm on `z` (§4.4), and the existence proof that
  if reconstruction ever becomes a capacity tax, dropping the decoder is safe *provided* the
  consistency+reward+value losses stay.
- **SPR → BBF** (2021 → 2023): the best *model-free* Atari-100k agents got there by adding
  latent **self-prediction as an auxiliary loss** — a predictor nobody plans with. Folded into
  §6.A: it de-risks P3 by giving the predictor a second payoff channel (auxiliary shaping of
  `V`/`π`) independent of whether planning ever ships.

### Stochastic line

- Direct follow-ups to Stochastic MuZero are thinner — the afterstate + discrete-chance-codebook
  factorization of 2022 is still the state of the art for discrete stochastic planning (2024–25
  extensions target continuous actions, offline settings, and belief-state variants — L-MAP,
  belief-aware MuZero — none of which we need). Two consequences: our §4.4 design is current, and
  the maintained implementation to borrow from is LightZero's Stochastic MuZero.

### Net design deltas adopted from this scan

1. §4.4: explicit **latent consistency loss** + **SimNorm/categorical latent normalization**.
2. §4.6 P-2: **Gumbel search** (sequential halving) instead of vanilla PUCT; budgets of 8–32
   sims are legitimate.
3. §6.A: predictor doubles as **SPR-style auxiliary loss** for the model-free head — P3 pays off
   even without a planner.
4. Implementation planning should evaluate **LightZero** as a base/reference before writing
   chance-node MCTS from scratch.
5. Keep the decoder predicting **absolute states** (x-prediction), never deltas.

## References

- MuZero: Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned
  Model*, Nature 2020.
- Stochastic MuZero: Antonoglou et al., *Planning in Stochastic Environments with a Learned
  Model*, ICLR 2022 — [paper](https://openreview.net/forum?id=X6D9bAHhBQ1),
  [author's summary](https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/).
- DreamerV3: Hafner et al., *Mastering Diverse Domains through World Models*, 2023.
- World Models: Ha & Schmidhuber, 2018. JEPA: LeCun, *A Path Towards Autonomous Machine
  Intelligence*, 2022.
- TD-Gammon: Tesauro, *Temporal Difference Learning and TD-Gammon*, CACM 1995 (afterstate values).
- Gumbel MuZero: Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022.
- EfficientZero: Ye et al., NeurIPS 2021; V2: [arXiv:2403.00564](https://arxiv.org/abs/2403.00564).
- UniZero: [arXiv:2406.10667](https://arxiv.org/abs/2406.10667); LightZero (framework):
  [arXiv:2310.08348](https://arxiv.org/abs/2310.08348), github.com/opendilab/LightZero.
- Dreamer 4: Hafner & Yan, *Training Agents Inside of Scalable World Models*,
  [arXiv:2509.24527](https://arxiv.org/abs/2509.24527).
- STORM: Zhang et al., NeurIPS 2023 (efficient stochastic transformer world model).
- TD-MPC2: Hansen et al., ICLR 2024 (consistency-only latent model, SimNorm).
- SPR: Schwarzer et al., ICLR 2021; BBF: Schwarzer et al., ICML 2023 (self-predictive aux losses).
- V-JEPA 2: [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) (action-conditioned JEPA planning).
- LLM agents on STS1 (context for Alternative D):
  [*Language-Driven Play: LLMs as Game-Playing Agents in Slay the Spire*](https://dl.acm.org/doi/fullHtml/10.1145/3649921.3650013).
