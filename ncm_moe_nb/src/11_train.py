# %% [markdown]
# ## 13. Train
#
# * One optimizer step = `ACCUM` micro-batches (each a single row type, a fixed token budget) -> ~`BATCH_CODES` live codes.
# * AdamW (fused), beta (0.9, 0.95), wd on matrices only, linear warmup then cosine to 10%, grad-clip, bf16 autocast.
# * Logged every `LOG_EVERY` steps: loss, per-(modality, level) CE, tokens/s, TFLOPS/MFU, and the **share of the step spent
#   waiting for data**, which tells a feed-bound run from a compute-bound one.
# * Held-out evaluation every `EVAL_EVERY_B` billion codes against the **constant-predictor baseline**: the cross-entropy of the
#   held-out codes under the *training* unigram of that (modality, level). A score under it is not a result.
# * Checkpoints (weights, optimizer, step, codes seen, codebooks) go to `CKPT_DIR` every `CKPT_MIN` minutes and a watchdog saves
#   and stops before `SESSION_HOURS`. Data is never saved; a resume re-streams with a fresh shuffle seed ("approximately resume").
# * Early abort: if the loss has not fallen 5% by step 200, stop instead of burning the budget.
# * **Caption dropout** (`CAP_DROP`): a fraction of image and audio rows train with an empty caption/transcript, so at sampling time
#   the model can be pushed away from "no caption" toward the caption (classifier-free guidance, `cfg` in the generate cell).

# %%
HIST = []
UNI = {}          # (modality, level) -> running histogram of TRAINING target codes, the honest constant predictor


def make_optimizer():
    dec, nodec = [], []
    for n, p_ in net.named_parameters():
        (dec if p_.ndim >= 2 and not n.startswith(("emb", "pos")) and "router" not in n else nodec).append(p_)
    return torch.optim.AdamW([dict(params=dec, weight_decay=WD), dict(params=nodec, weight_decay=0.0)],
                             lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=IS_CUDA)


WARM = min(WARMUP, max(1, STEPS_TOTAL // 5))
def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    prog = min(1.0, (step - WARM) / max(1, STEPS_TOTAL - WARM))
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate():
    """Teacher-forced held-out CE / top-1 per (modality, level) + the constant-predictor baseline."""
    net.eval()
    acc = {}
    for mb in VAL:
        r = mb["rtype"]; m = ROW_MOD[r]
        with _amp():
            h = net.hidden(r, mb["cond"], mb["codes"])
        for k in range(LAYOUTS[r].K):
            lg = net.level_logits(r, h, k)
            tg = mb["codes"][k]
            ce = F.cross_entropy(lg.reshape(-1, SPEC.V), tg.reshape(-1), reduction="none")
            a = acc.setdefault((m, k), dict(ce=0.0, ok=0.0, n=0, cnt=torch.zeros(SPEC.V, device=dev)))
            a["ce"] += float(ce.sum()); a["ok"] += float((lg.argmax(-1) == tg).sum()); a["n"] += tg.numel()
            a["cnt"] += torch.bincount(tg.reshape(-1), minlength=SPEC.V).float()
    out = {}
    for (m, k), a in acc.items():
        fv = a["cnt"] / a["cnt"].sum()                              # held-out code frequencies
        u = UNI.get((m, k))
        if u is not None and float(u.sum()) > 0:                    # constant predictor = training unigram, add-one smoothed
            lp = ((u + 1) / (u.sum() + SPEC.V)).log()
            base = float(-(fv * lp).sum())
        else:                                                       # nothing trained yet: held-out entropy (optimistic)
            base = float(-(fv[fv > 0] * fv[fv > 0].log()).sum())
        out[f"{m}_L{k}"] = dict(ce=a["ce"] / a["n"], acc=a["ok"] / a["n"], base=base, n=a["n"])
    net.train()
    return out


BEST = {"score": float("inf")}      # best harness-wide held-out CE so far (set in the eval block)


def save_ckpt(step, opt, seen, path=None):
    if not CKPT_DIR:
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = path or CKPT_PATH
    tmp = path + ".tmp"
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, seen=seen, hist=HIST[-2000:],
                    codecs=codecs.state(), preset=PRESET), tmp)
    os.replace(tmp, path)
    print(f"  checkpoint -> {path} (step {step})", flush=True)


def fmt_val(v):
    mods = {}
    for key, d in v.items():
        mods.setdefault(key.split("_")[0], []).append(d)
    return " | ".join(f"{m} ce {np.mean([d['ce'] for d in ds]):.2f} (base {np.mean([d['base'] for d in ds]):.2f}, "
                      f"acc {100*np.mean([d['acc'] for d in ds]):.0f}%)" for m, ds in mods.items())


def train():
    global HIST
    net.train()
    opt = make_optimizer()
    seen, step0 = dict(text=0, image=0, audio=0), 0
    if RESUME is not None:
        net.load_state_dict(RESUME["net"]); opt.load_state_dict(RESUME["opt"])
        step0, seen, HIST = RESUME["step"], RESUME["seen"], RESUME["hist"]
        print(f"resumed at step {step0}, codes seen {seen}")
    eval_every = max(1, min(round(EVAL_EVERY_B * 1e9 / CODES_PER_STEP), max(1, STEPS_TOTAL // 3)))
    writer, stream = None, None
    if SHARD_MODE == "read" and SHARD_DIR:
        pf = ShardFeeder(SHARD_DIR, start_micro=step0 * ACCUM, seed=SEED, micro_left=(STEPS_TOTAL - step0) * ACCUM)
    else:
        if SHARD_MODE == "record" and SHARD_DIR:
            os.makedirs(SHARD_DIR, exist_ok=True)
            need, free = (STEPS_TOTAL - step0) * CODES_PER_STEP * 1.4 / 1e9, shutil.disk_usage(SHARD_DIR).free / 1e9
            if need > 0.9 * free:
                print(f"  !! shards would need ~{need:.0f} GB but only {free:.0f} GB are free: not recording this run")
            else:
                writer = ShardWriter(SHARD_DIR)
                print(f"recording shards to {SHARD_DIR} (~{need:.1f} GB for this run, {free:.0f} GB free)")
        ds = MixDataset(SPEC, part0=1, nparts=PARTS, seed=SEED + 1 + step0)
        stream = RowStream(ds, WORKERS)
        pf = Prefetcher(stream, codecs, writer=writer)
        print(f"streaming with {WORKERS} workers; first batch can take a minute while sources connect...", flush=True)
    t_run, t_ckpt, t_log, wait, steps_in_log = time.time(), time.time(), time.time(), 0.0, 0
    lvl_sum, lvl_n, loss_sum = {}, {}, torch.zeros((), device=dev)
    live_in_log, packed_in_log = 0, 0
    abort_at = 200 if STEPS_TOTAL >= 400 else None
    try:
        for step in range(step0, STEPS_TOTAL):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            for _ in range(ACCUM):
                t0 = time.time()
                mb = pf.get()
                wait += time.time() - t0
                cond = mb["cond"]
                if CAP_DROP > 0 and cond is not None and mb["rtype"] in ("image", "audio"):
                    blank = torch.rand(cond.shape[0], 1, device=cond.device) < CAP_DROP       # blank the caption/transcript in a fraction of rows,
                    cond = torch.where(blank, torch.ones_like(cond), cond)                      # so the model also learns "no caption" (= all padding, code 1)
                with _amp():
                    loss, per_level = loss_fn(mb["rtype"], cond, mb["codes"])
                (loss / ACCUM).backward()
                m = ROW_MOD[mb["rtype"]]
                loss_sum += loss.detach() / ACCUM
                lvl_sum[(m, mb["rtype"])] = lvl_sum.get((m, mb["rtype"]), 0) + per_level
                lvl_n[(m, mb["rtype"])] = lvl_n.get((m, mb["rtype"]), 0) + 1
                seen[m] += mb["n_live"]; live_in_log += mb["n_live"]
                for k_, c_ in enumerate(mb["codes"]):
                    UNI[(m, k_)] = UNI.get((m, k_), 0) + torch.bincount(c_.reshape(-1), minlength=SPEC.V).float()
                packed_in_log += len(mb["src"]) * LAYOUTS[mb["rtype"]].T
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP, foreach=True)
            opt.step()
            steps_in_log += 1
            done = step + 1
            if done % LOG_EVERY == 0 or done == STEPS_TOTAL or done == 1:
                if IS_CUDA: torch.cuda.synchronize()
                dt = time.time() - t_log
                loss_v = float(loss_sum) / steps_in_log
                ce = {}
                for (m, r), v in lvl_sum.items():
                    for k, x in enumerate((v / lvl_n[(m, r)]).tolist()):
                        ce.setdefault(f"{m}_L{k}", []).append(x)
                ce = {k: round(float(np.mean(v)), 4) for k, v in ce.items()}
                tok_s = packed_in_log / dt
                tfl = 6 * REPORT["active_max"] * tok_s / 1e12
                rec = dict(step=done, loss=loss_v, lr=lr_at(step), gnorm=float(gn), codes_s=live_in_log / dt,
                           tflops=tfl, wait=wait / dt, seen=dict(seen), ce=ce)
                HIST.append(rec)
                eta = (STEPS_TOTAL - done) * dt / steps_in_log / 3600
                mfu = f" ({100*tfl/PEAK_TFLOPS:.0f}% MFU)" if PEAK_TFLOPS else ""
                print(f"step {done}/{STEPS_TOTAL} | loss {loss_v:.3f} | lr {rec['lr']:.1e} | gn {rec['gnorm']:.2f} | "
                      f"{rec['codes_s']/1e3:.1f}k codes/s | {tfl:.1f} TFLOPS{mfu} | data-wait {100*rec['wait']:.0f}% | "
                      f"seen {sum(seen.values())/1e9:.4f}B | eta {eta:.1f}h", flush=True)
                if rec["wait"] > 0.5 and done > 3 and (done // LOG_EVERY) % 10 == 1:
                    print(f"  !! feed-bound: the GPU waits {100*rec['wait']:.0f}% of each step for data. "
                          f"Raise WORKERS (now {WORKERS}) or run where more CPU cores / bandwidth are available.", flush=True)
                lvl_sum, lvl_n, loss_sum = {}, {}, torch.zeros((), device=dev)
                live_in_log = packed_in_log = steps_in_log = 0
                wait, t_log = 0.0, time.time()
                if not math.isfinite(loss_v):
                    raise FloatingPointError("non-finite loss: stop and lower LR")
            if abort_at and done == abort_at:
                tr = [h["loss"] for h in HIST if "loss" in h and h["step"] <= done]   # val records have no "loss"
                if len(tr) >= 8 and np.mean(tr[-3:]) > 0.95 * np.mean(tr[:3]):
                    print("NOT LEARNING: loss fell <5% in 200 steps. Stop and check the CE table before spending the budget.")
                    print(HIST[-1]["ce"])
                    return False
            if done % eval_every == 0 or done == STEPS_TOTAL:
                v = evaluate()
                HIST.append(dict(step=done, val=v))
                print(f"  eval @ {done}: {fmt_val(v)}", flush=True)
                # Keep the best held-out model in its own file. The rolling checkpoint is
                # overwritten every CKPT_MIN minutes, which is how audio's best model (held-out
                # 1.52 at 0.73B codes) was replaced by a late overfit 1.78 as the only copy.
                sc = float(np.mean([d["ce"] for d in v.values()]))
                if sc < BEST["score"]:
                    BEST["score"] = sc
                    _best = CKPT_PATH.replace(".pt", "_best.pt")
                    save_ckpt(done, opt, seen, path=_best)
                    print(f"  best held-out so far: mean CE {sc:.3f} -> {_best}", flush=True)
            if CKPT_DIR and time.time() - t_ckpt > CKPT_MIN * 60:
                if writer is not None:
                    writer.flush_all()
                save_ckpt(done, opt, seen); t_ckpt = time.time()
            if SESSION_HOURS and time.time() - t_run > SESSION_HOURS * 3600:
                save_ckpt(done, opt, seen)
                print(f"watchdog: {SESSION_HOURS} h reached at step {done}. Checkpoint saved: re-run this notebook with the same "
                      f"CKPT_DIR to resume.")
                return True
        save_ckpt(STEPS_TOTAL, opt, seen)
    finally:
        pf.close()
        if stream is not None:
            stream.close()
        if writer is not None:
            writer.close()
    return True


if not META_ONLY and not EVAL_ONLY:                 # EVAL_ONLY: the eval notebook defines evaluate() and the helpers above but never trains
    TRAINED = train()
