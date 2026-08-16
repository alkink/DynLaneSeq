# V25 Sol-Pro component gate plan

## Objective

Build the complete image-mediated four-lane object detector proposed in the
Sol-Pro review, while testing each separable mechanism before committing a
50-epoch official CULane run.

The gates are falsification checks, not model selection sweeps.  They use one
fixed seed, one fixed endpoint, the untouched official `list/train.txt`, and
the complete official `list/val.txt`.  No image, row, clip, duplicate, or hard
case is removed from either population.

## Non-negotiable data/evaluation contract

- Training population: every row of official `list/train.txt`.
- Validation population: every row of official `list/val.txt`.
- No `train_gt.txt`, deduplication, clip filtering, or validation subset.
- Short learned gates consume exactly one complete train-list epoch unless a
  gate is explicitly marked training-free.
- Every learned arm starts from the same initialization and sees the same
  seeded augmentation/sample stream as its paired control.
- Every scientific result uses the full official validation population and a
  fixed endpoint.  There is no checkpoint, threshold, NMS, or seed sweep.
- Test remains closed until the complete model improves full validation by at
  least 0.8 F1 points at IoU .50 without regressing IoU .75.

## What a short gate can and cannot prove

A short gate can expose broken tensor semantics, missing gradients, collapse,
wrong image dependence, destructive competition, or a component that is
already materially harmful.  It cannot prove the final 50-epoch capacity of a
new architecture.  Failure to beat V7 after one epoch is therefore not an
automatic stop; failure of the component's own causal/mechanical contract is.

## Fixed component sequence

### M0 — Mechanical and oracle contract (no training)

Implement the full configurable graph and require:

- four final lane objects are the only writer geometry owner;
- no V7/source geometry, global gate, source/raw router, or proposal ID is
  present in the deployment graph;
- canonical target ordering, empty-lane handling, range and existence tensors
  are shape- and permutation-correct;
- every enabled trainable block receives finite nonzero gradient;
- an exact-GT unary cost volume is recovered by hard path decoding;
- proposal memory cannot constrain final x to the proposal convex hull;
- swapping/copying image evidence changes the image-owned output;
- official train/val list counts and hashes are recorded.

### G0 — Direct four-lane image-object core (one official epoch)

Enabled:

- ImageNet-initialized trainable encoder and high-resolution FPN;
- four canonical lane-row objects;
- full-width row image attention/cost volume;
- direct row distribution, existence, range, and quality supervision;
- vertical intra-lane reasoning;
- soft structured path training and hard coherent path inference.

Disabled:

- auxiliary proposals and proposal prior;
- image-mediated cross-lane coverage competition;
- V7-hard reweighting and denoising queries.

Required mechanism evidence:

- train row/path loss decreases by at least 20%;
- correct-image geometry beats deterministic cross-clip wrong-image geometry;
- hard path has no more crossings/fragmentation than independent row argmax;
- output is not collapsed to fewer than two active lane objects over the full
  validation population.

### G1 — Hard coherent path (training-free paired replay)

On the fixed G0 endpoint compare, from identical logits:

1. row expectation;
2. independent row argmax;
3. whole-curve MAP/Viterbi path.

The path arm is retained when it improves continuity/crossing and does not
materially regress official F1@.50.  A failed exact-GT path oracle is an
implementation bug, not a model result.

### G2 — Image-mediated four-lane ownership competition

Clone the fixed G0 endpoint into control and treatment and continue both for
the same quarter-epoch stream.  Treatment enables explicit competition over
the image ridge ownership maps; control leaves the four objects independent.

Required mechanism evidence:

- duplicate-ridge and crossing rates decrease;
- unique GT coverage does not decrease;
- correct-image advantage is retained;
- full-validation F1@.50 and F1@.75 are not materially worse than control.

### G1B — Diverse coherent-path capacity (training-free)

From the same fixed G0 unary tensor, extract the exact MAP path and two
suppression-diversified coherent alternatives per final slot.  Compare the
single-MAP writer, deterministic `(K + dustbin)^4` energy/set decode, and a
GT-only capacity oracle.  The oracle is diagnostic and is never deployed.

Required evidence before a learned multi-path selector is considered:

- path 0 is bit-exact with the G1 hard-Viterbi result;
- alternative paths are spatially distinct rather than numerical copies;
- deterministic set decode does not materially regress single-MAP F1;
- multi-path official oracle adds at least `+0.30` F1@.50 to justify retaining
  the extra hypotheses, and at least `+0.80` to authorize a learned selector.

### G3 — Auxiliary proposals, dual energy, and multi-path set decode

First test the auxiliary branch alone for proposal coverage.  Then continue a
paired control/treatment for the same quarter epoch.  Both arms train the same
32-proposal one-to-many coverage branch and row-visibility/reliability heads.
Control keeps image-only final energy; treatment enables a calibrated
log-mixture of separate image and proposal spatial energies. Proposal tensors
enter the final branch only as a droppable spatial prior/memory. They never own
writer coordinates and never form a coordinate weighted average. Inference
preserves three coherent paths per final lane and performs an exact
`(K + dustbin)^4` deterministic set decode from the trained spatial energy,
existence evidence, and explicit order/duplicate costs.

Required mechanism evidence:

- auxiliary proposal oracle coverage is above the four-object output;
- zeroing proposal memory does not collapse the image-only final detector;
- a synthetic target outside the proposal cloud remains reachable;
- proposal treatment does not reduce unique final-lane coverage or official
  validation F1 relative to its paired control.
- image/proposal modes remain spatially distinct when they disagree; no
  coordinate expectation is allowed between them;
- final writer paths are exact members of the coherent hypothesis bank;
- row-level visibility and entropy/margin-aware q50/q75 heads receive finite
  nonzero gradients from the same final lane-object state.

### G4 — Training-only denoising lane queries

Add GT/proposal paths perturbed by fixed horizontal shifts, range noise, row
dropout, and occlusion.  They share the final decoder weights and never appear
at inference.  A paired quarter-epoch gate must show improved correction of
held-out perturbations without increasing clean-lane error.

### G5 — Official-population tail emphasis

Every official training row remains in every epoch.  Tail emphasis is applied
as a capped per-sample/per-lane loss weight (and is reported), not by deleting
or replacing the official population.  A paired quarter-epoch gate must
improve V7-hard validation lanes without materially regressing the full
population.

### G6 — Complete Sol-Pro model

Combine every mechanism that passed its own causal contract:

- trainable pretrained image encoder;
- high-resolution full-width evidence;
- 32 one-to-many proposals as auxiliary coverage memory;
- four one-to-one final lane-row objects as the only output owner;
- vertical intra-lane reasoning and coherent hard path inference;
- separate image/proposal energy distributions and calibrated log-mixture;
- multiple diverse coherent path hypotheses with exact small-set decode;
- image-mediated ridge ownership competition;
- row-level visibility and posterior-aware q50/q75 reliability;
- deterministic order/non-crossing/duplicate constraints;
- proposal/GT denoising supervision;
- direct geometry, range, existence, quality, soft strip-IoU, and candidate
  hard-negative supervision;
- transparent capped tail emphasis over the unchanged official population.

Run the fixed 50-epoch schedule only after M0 and G0–G5 complete.  Use one
predeclared final endpoint.  Do not add a post-hoc V7 gate or router if the
complete detector fails.

## Interpretation rule

The core four-lane object contract (G0) is the hypothesis under test and is not
discarded merely because one epoch is below V7.  Optional mechanisms G2–G5
must earn inclusion by their causal contract.  If an optional mechanism fails,
fix its semantics and rerun the same predeclared gate; do not compensate with
threshold or loss-weight sweeps.
