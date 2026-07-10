# Session handoff — ruffle (2026-07-04)

Untracked scratch file for the next session. Do not commit it; delete when absorbed.
Written assuming PRs #8, #9, #10 have been merged by the user; verify with
`git fetch origin main` and `gh pr list` before acting, and adjust if any is still open.

## 1. Repo state (github.com/lathrys-at/ruffle)

PRs opened this session, assumed landed on `main`:

- #8 `feat/tuned-defaults`: retuned discrimination defaults (top_eps 0.10, top_m 5,
  winsor_z 2.5, denom_floor_frac 0.75), docs sweep, fixture regen.
- #10 `feat/base-weight`: `ChannelConfig.base_weight` in crate + both bindings,
  CHANGELOG, tuning-guide section with floor-not-zero advice.
- #9 `feat/eval-large-corpora`: heavy runners (cqadupstack, msmarco), dense channel
  = gte-modernbert-base, delta profiles, SUMMARY.md generator, fitted conditions
  (`ruffle_evals/fitted.py`) and results for scifact/nfcorpus/fiqa/quora/cqadupstack.
  results/msmarco.json intentionally absent pending its gte rerun.

After all three land: local `main` needs a pull; the local working tree sits on
`feat/eval-large-corpora` (delete or leave after merge); worktree `/tmp/bw-wt`
(`feat/base-weight`) is disposable. #8 and #10 both touch CHANGELOG under
[Unreleased]; whichever merges second may need the trivial both-sections resolution
(an Added and a Changed section, keep both; resolved once already in this session's
throwaway wheel merge). The eval venv (`evals/.venv`) wheel was built from exactly
that merge, so once both land a wheel built from `main` is equivalent; no reinstall
needed unless main drifts further.

Still LOCAL ONLY: branch `docs/roadmap` (worktree `/tmp/roadmap-wt`) carrying
`docs/roadmap.md` and the draft `docs/proposals/feedback-learned-weights.md`
(commit `f416e49`). After #10 lands, the roadmap's own convention says the
base_weight item moves out of "Planned for 0.3" (it shipped into the changelog);
do that pruning in the same commit as the proposal revision. Rebase the branch onto
updated main before pushing.

The user opens and merges PRs themselves. Never push main, never force-push
(user-only; the permission classifier blocks it regardless).

## 2. Running: MS MARCO gte rerun (the long pole)

Background since Jul 3 ~21:00: embedding 8.84M passages at ~54/s (log:
`evals/cache/gte-full.log`, lines `[msmarco] embedded N/8841823`; 2.64M at Jul 4
10:30). ETA: embedding done Jul 5 evening, then 1–3 h fusion. The process holds old
harness code in memory. On completion:

1. Sanity-check `results/msmarco.json`: `engine_defaults` present, cold == rrf
   exactly, dev/dl19/dl20 plausible for gte-modernbert.
2. Run `evals/.venv/bin/python -m ruffle_evals.summarize` manually (the in-memory
   CLI regenerates RESULTS.md but not SUMMARY.md for this run).
3. Commit the msmarco results on a NEW branch off updated main (eval branch is
   merged under the assumption above) and open a small follow-up PR.
4. msmarco lacks the fitted conditions: extend `heavy.py::run_msmarco` (fit on the
   dev warm split, evaluate via the `ruffle_warm_multi`/resume path), rerun msmarco
   fusion-only from caches (~1–2 h; embeddings and run caches are keyed and present),
   include in the same follow-up PR.

Monitoring was via ScheduleWakeup heartbeats; re-establish if resuming before it
finishes. If the process died: everything else is banked per-collection; the msmarco
embedding memmap has NO resume (restart = re-embed ~45 h; warn the user first and
consider adding resume + fp16 before restarting).

## 3. Published findings of record (results/SUMMARY.md on the merged branch)

- Ruffle warm is the only method above RRF in every column; ISR/CombSUM beat it on
  dense-dominant collections; dense alone beats all label-free fusion on
  fiqa/quora/cqadupstack; the oracle bracket is tight on balanced collections.
- Fitted weights: small budgets recover most of the oracle gap (fiqa: oracle exactly
  from 16 graded queries on 2/3 draws; cqadupstack pooled ~90–93%). Composing with
  adaptation rescues bad NON-ZERO fits; a fitted zero silences a channel
  irrecoverably (nfcorpus draw 0 = 0.3255, below the RRF floor, shown in the table).
  CAVEAT from review: composed rows carry real p5 loss tails (−0.20…−0.37) vs plain
  warm Ruffle's ~0.00; do not oversell robustness anywhere.

## 4. Active task: feedback-learned weights proposal (user-commissioned)

Arc: user asked how ruffle could approach oracle quality given "feedback through
some means" → draft proposal written (tiers: explicit feedback API, active grading
via the conflict diagnostic, implicit feedback excluded) + roadmap entry → user
requested 3 Opus adversarial reviewers against draft+roadmap → incorporate changes
ALL agree on → post the finalized design as a GitHub issue along with the roadmap.

Review status:
- `critic-peer`: DELIVERED 12 findings (2 blocking): (1) the draft's robustness
  claim contradicts SUMMARY's own loss tails; the eval plan never gates on
  do-no-harm; (2) an offline fit helper emitting base_weight (no state change)
  achieves everything the data proves; the state format bump is unjustified as
  written. Also: recovery percentages overstate first-draw numbers; active grading
  is hypothesis-as-design; product of three floored factors is weaker than the
  documented single floor (floor the composed weight); "additive not revision" is
  partly marketing; several style violations quoted.
- `critic-stats`: DELIVERED 14 findings (2 blocking): (1) credit is per-channel-
  absolute while oracle weights are joint, so redundancy inverts the learned tilt
  (needs a designed redundant-pair condition; consider leave-one-out credit);
  (2) conflict-selected grading is biased sampling that can invert the traffic-
  optimal tilt (train-on-conflict / test-on-representative required). Cluster:
  normalize credit within each event before pooling; MeanVar is sufficient for only
  1 of 3 candidate estimators (bake-off before persistence design); absent doc = 0
  conflates quality with list depth; absent CHANNEL must not count as zero; decay
  must be event-clocked, not query-clocked; shrink in log space; support graded
  relevance, not only binary; merge pools population-dependent magnitudes.
- `critic-systems`: NEVER REPORTED. Nudge via SendMessage (name: critic-systems) or
  re-spawn with the same mandate (systems/API/format-bump/parity attack; pointed at
  fuser.rs, state.rs, keys.rs, config.rs, CHANGELOG). Full reviewer outputs are in
  the prior session transcript:
  /Users/lupine/.claude/projects/-Users-lupine-Development-ruffle/3e229f43-9e90-4f9e-8372-befd36ab95d9.jsonl

Consensus so far (both delivered reviews agree; expect systems to concur): (a) no
persisted statistic / format bump before an estimator bake-off; (b) gate the
active-grading guidance on the conflict-vs-random experiment; (c) event-clocked
decay; (d) motivation must cite SUMMARY's actual numbers and tails; (e) floor the
composed weight, not each factor.

User additions to fold into the revision (from conversation, not yet in the draft):
- Feedback primitive is STATE-LEVEL and pure (state + events → new state);
  `Fuser.feedback` is a thin live wrapper; third CLI subcommand in the existing
  reconcile/rekey pattern: `ruffle feedback --state in.json --events graded.jsonl
  --out out.json`. Composes with merge for central-grading → reconcile-out flows.
- Operating-guide loop for conflict telemetry: retain (query, conflict, per-channel
  ranked ids) above a conflict threshold or in a top-N ring buffer at query time
  (feedback events need the rank lists), grade offline, feed through the CLI. Ship
  the guidance only with the acquisition experiment's verdict.

Likely restructure (consistent with all input): staged proposal. Stage 1: offline
fit helper + base_weight, no engine change. Stage 2: harness bake-off with named
conditions (redundant dense pair, conflict-vs-representative shift, difficulty
sensitivity, depth invariance, sparse-event decay, label noise, do-no-harm tail
gate vs warm Ruffle). Stage 3: the streaming layer only if stage 2 picks a
surviving estimator; format bump justified then; persisted statistic chosen by the
bake-off, not MeanVar by default.

Completion steps: revise proposal + prune roadmap in `/tmp/roadmap-wt`, commit,
rebase onto updated main, push `docs/roadmap`, post the finalized design as a
GitHub issue (`gh issue create`) including the roadmap context. Issue rules: no
Claude-Session links, no instructions addressed to the user, plain prose, no
em-dashes.

## 5. Standing constraints (memory-backed; do not violate)

- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; PR
  bodies end with the "Generated with Claude Code" line; NEVER a Claude-Session
  link in either.
- Never push main; never force-push; user creates tags/releases and merges PRs.
- Flaky degraded-mode numbers: experiment keeps running but its numbers are never
  cited in tuning decisions or public prose (memory: flaky-not-evidence).
- Writing style everywhere: plain direct prose, no em-dashes, no antithesis/
  aphorisms/meta-commentary, descriptive headings, "Ruffle" capitalized in prose
  (memory: writing-style-no-llm-ese; ~/.claude/skills/doc-cleanup).
- Background shell commands: absolute paths only (cwd drift bit us twice).

## 6. Parked / deferred (documented, not commissioned)

- Qwen3-Embedding-0.6B rerun: blocked on memmap resume + fp16 (evals/README.md
  Deferred section).
- Heterogeneous-channel / multi-modal-proxy evaluation (roadmap candidate).
- Second dense channel for the coupling estimator (roadmap candidate).
- Floated, not commissioned: ISR/CombSUM under the degraded experiment; eta (RRF
  discount) search with reversed-split hygiene.
- Cleanup when idle: /tmp/bw-test-venv; /tmp/bw-wt worktree and local branch
  tmp leftovers (untracked wheels/ dir inside it); local feature branches after
  their PRs merge.
