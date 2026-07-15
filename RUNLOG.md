# RUNLOG — 2,000 Step LLM Speedrun

Two runs have console output to draw real numbers from. Only these two are
documented below — I didn't invent extra entries to pad this out. If you ran
more experiments in between (different `n_embd`, different `max_chars` on the
tokenizer, dropout sweeps, etc.), add them as Run 1.5 / 2.5 with their real
numbers; happy to fold them in.

---

## Run 1 — compact char-fallback tokenizer + RMSNorm/SwiGLU

**Command:** `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`
(no explicit `--micro_batch/--accum_steps/--lr`, so this used whatever
defaults that version of `train.py` had at the time)

**Hypothesis:** A larger character vocabulary (byte-fallback + top-K frequent
chars) should shorten sequences relative to raw UTF-8 bytes, giving the model
an easier per-step prediction target than byte-level modeling while staying
well under the 2M param cap.

**What changed vs. starter:** tokenizer upgraded from raw byte-level (vocab
256) to a char+byte-fallback tokenizer; model upgraded from vanilla
attention/MLP to RMSNorm + SwiGLU. *(Note: this run's exact `model.py` isn't
among the uploaded files — the params/vocab numbers below are what the log
shows, but I can't reconstruct the exact layer widths used. Worth keeping a
copy of every `model.py` variant you train, tagged by run, next time.)*

| Metric | Value |
|---|---|
| Params | 1,424,800 (71.2% of 2M cap) |
| Vocab size | 1,024 (256 byte fallback + 768 learned chars) |
| Corpus | 7,318,592 bytes → 2,628,712 tokens (2.78 bytes/token) |
| Final train loss (step 2000) | 4.0761 |
| Dev bpb | **not measured** — no `evaluate.py` call appears in this session's log for this checkpoint |

**Conclusion:** Loss dropped smoothly from 6.95 → 4.08, so training is stable,
but a train loss of ~4 in a ~1024-way vocab isn't directly comparable to
anything else — that's exactly why the assignment scores bpb, not loss.
Without a dev bpb number this run is a reference point for "training works,"
not a scored baseline. **Action item:** if you still have this checkpoint,
run `python evaluate.py --checkpoint <this_ckpt> --text_file ../data/dev_eval.txt`
and drop the number in here before submitting.

---

## Run 2 — LLaMA-style RoPE architecture + gradient-accumulation trainer (final submission)

**Command:** `python -u train.py --data ../data/train_corpus.txt --steps 2000 --micro_batch 8 --accum_steps 2 --lr 8e-4 --out ckpt.pt`

**Hypothesis:** Swapping learned absolute position embeddings for RoPE, widening
the model to use more of the 2M param budget (192-dim, 6 heads, 4 layers,
256 context), and using gradient accumulation (effective batch 16) with a
OneCycleLR schedule tuned for a known, fixed step count should converge faster
and generalize better than Run 1's smaller config, within the same 2,000-step
cap.

**What changed vs. Run 1:** architecture is now the uploaded `model.py`
(RMSNorm, SwiGLU, RoPE, tied embeddings, `block_size=256`, `n_embd=192`,
`n_head=6`, `n_layer=4`); trainer adds OneCycleLR super-convergence and
label smoothing (0.05); tokenizer vocab differs (913 vs 1,024 — see caveat
below).

| Metric | Value |
|---|---|
| Params | 1,946,496 (97.3% of 2M cap) |
| Vocab size | 913 (256 byte fallback + **657** learned chars) |
| Corpus | 7,318,592 bytes → 5,703,936 tokens (1.28 bytes/token) |
| Final train loss (step 2000) | 1.4286 |
| Wall clock | 1,876 s (~31.3 min) for 2,000 steps, ~940–1,150 ms/step |
| **Dev bpb (measured)** | **1.801** |
| Tokens in eval file | 112,854 (112,853 scored) |

