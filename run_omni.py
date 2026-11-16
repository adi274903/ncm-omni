"""One entry point for the whole omni pipeline: config -> data -> train -> val -> eval.

    python run_omni.py check                     # invariants, no data needed
    python run_omni.py produce  [--budget N]     # writes cfg/*.json then runs the producer
    python run_omni.py train --smoke 10          # timing run, writes nothing
    python run_omni.py train                     # full run + checkpoints
    python run_omni.py eval                      # NFE-vs-quality table
    python run_omni.py all --budget 20000        # produce in the background, wait, train, eval

Both arms use the same schedule, vocab and model width; the only difference is the attention
mask (`block` vs `ar`), which is what makes the comparison a matched-params comparison.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")


# Model/optimiser presets. Parameter counts are measured, not nominal: at base_len=1024 the
# positional table is 1025 x d, so the same width counts for more than it does at 64.
#   sm  10.44M  d384  L6  h6  kv2     -> the debug harness that produced the 10M result
#   md  83.6M   d768  L13 h12 kv3     -> the kill test rung (measured 83.59M at base_len 1024)
#   lg  454.1M  d1536 L18 h16 kv4     -> the paper number, rented H100 only
#   moex 1.178B total / 459M active   d1536 L16 h16 kv4, 3 experts x 359.8M -> needs
#                                      --moe 3 --moe-ffn 4880, or it is just another lg
PRESETS = {
    "sm": dict(n_embd=384,  n_layer=6,  n_head=6,  n_kv_head=2, batch=8,  lr=3e-4,   warmup=100),
    "md": dict(n_embd=768,  n_layer=13, n_head=12, n_kv_head=3, batch=8,  lr=2e-4,   warmup=200),
    "lg": dict(n_embd=1536, n_layer=18, n_head=16, n_kv_head=4, batch=8, lr=1.5e-4, warmup=300),
    "moex": dict(n_embd=1536, n_layer=16, n_head=16, n_kv_head=4, batch=8, lr=1.5e-4, warmup=300),
}


def _parse_r(spec):
    """'2' -> 2; 'text=2,image=4,audio=4' -> {'text': 2, 'image': 4, 'audio': 4}.

    Per-modality R is what lets text have a coarse-safe ladder (R=2) while image and audio
    keep R=4. The shared ratio is what gave text ONE level-0 code for 256 bytes in the K=5/R=4
    ladder, and a level-0 code that stands for the whole sample carries nothing learnable.
    """
    from omni.schedule import MODALITIES          # local: this module imports lazily
    spec = str(spec).strip()
    if "=" not in spec:
        return int(spec)
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k not in MODALITIES:
            raise SystemExit(f"--r: unknown modality {k!r}; pick from {MODALITIES}")
        out[k] = int(v)
    return out


def apply_overrides(cfg, a):
    """Preset dims, schedule length, eval-set size. Every command goes through here, because
    each one rewrites run/cfg/*.json - a mismatch between produce and train would silently
    train on a schedule the shards were not encoded with."""
    preset = getattr(a, "preset", None)
    if preset and preset in PRESETS:
        p = PRESETS[preset]
        cfg["model"].update({k: p[k] for k in ("n_embd", "n_layer", "n_head", "n_kv_head")})
        cfg["train"].update(batch=p["batch"], lr=p["lr"], warmup=p["warmup"])
    if getattr(a, "batch", 0):
        cfg["train"]["batch"] = int(a.batch)
    if getattr(a, "no_compile", False):
        cfg["train"]["compile"] = False
    if getattr(a, "save_every", None):
        cfg["train"]["save_every"] = int(a.save_every)
    if getattr(a, "log_every", None):
        cfg["train"]["log_every"] = int(a.log_every)
    n = getattr(a, "base_len", 0)
    if n:
        r = n / 64.0                       # 64 is the reference text/image length
        cfg["sched"]["base_len"] = dict(text=int(n), image=int(n),
                                        audio=max(4, int(round(96 * r / 4)) * 4),
                                        video=32)
        # video stays at 32: its base_len is frames x grid^2, so 512 would read 128 frames of
        # every clip. It is data-path only and currently dead in eval, so it is not worth the
        # decode. Raise it in build_codec if that ever changes.
    rspec = getattr(a, "r", None)
    if rspec:
        rs = _parse_r(rspec)
        if isinstance(rs, dict):
            cur = cfg["sched"]["R"]
            merged = dict(cur) if isinstance(cur, dict) else {}
            merged.update(rs)
            cfg["sched"]["R"] = merged
        else:
            cfg["sched"]["R"] = rs
    moe = getattr(a, "moe", 0)
    if moe:
        cfg["model"]["n_experts"] = int(moe)
        cfg["model"]["expert_map"] = {"text": 0, "image": 1, "audio": 2}
        mff = getattr(a, "moe_ffn", 0)
        if mff:
            cfg["model"]["moe_ffn"] = int(mff)
    hv = getattr(a, "hetero_vocab", 0)
    if hv:
        K = cfg["sched"]["K"]
        v = dict(cfg["vocab"])
        v["text"] = [int(cfg["vocab"]["text"])] * (K - 1) + [int(hv)]
        cfg["vocab"] = v
    sb = getattr(a, "shuffle_buf", 0)
    if sb:
        cfg["produce"]["shuffle_buf"] = int(sb)
    ev = getattr(a, "eval_every", None)
    if ev:
        cfg["train"]["eval_every"] = int(ev)
    pflr = getattr(a, "preflight_lr", None)
    if pflr is not None:
        cfg.setdefault("preflight", {})["lr"] = float(pflr)
    pfs = getattr(a, "preflight_steps", None)
    if pfs is not None:
        cfg.setdefault("preflight", {})["steps"] = int(pfs)
    if getattr(a, "no_preflight", False):
        cfg.setdefault("preflight", {})["enabled"] = False
    m = getattr(a, "holdout_mult", 0)
    if m:
        cfg["produce"]["holdout"] = {k: int(v * m) for k, v in cfg["produce"]["holdout"].items()}
    return cfg


def write_cfgs_all(cfg, budget=None, steps=None):
    return write_cfgs(cfg, budget=budget, steps=steps)


def default_cfg(budget=20000, out=os.path.join(RUN, "stream")):
    return dict(
        # three modalities: video was dropped because a frame is an image, so frames are encoded
        # as image samples instead of training a modality that measured below its copy prior.
        sched=dict(modalities=["text", "image", "audio"],
                   base_len=dict(text=64, image=64, audio=96), K=6, R=2, null_code=0),
        vocab=dict(text=257, image=257, audio=257),   # codebook_size + the NULL slot
        model=dict(n_embd=384, n_layer=6, n_head=6, n_kv_head=2, dropout=0.0,
                   qk_norm=True, mask_mode="block"),
        train=dict(steps=2000, batch=8, lr=3e-4, warmup=100, wd=0.01, clip=1.0,
                   log_every=10, eval_every=1000, save_every=500, compile=True),
        produce=dict(out_dir=out, sample_budget=budget, shard_samples=64,
                     max_ahead_shards=24, shuffle_buf=512, batch_docs=8, seed=3407,
                     holdout=dict(text=32, image=24, audio=16),
                     # 41.7 / 41.7 / 16.7 of SAMPLES, which is equal LIVE CODES per modality at the
                     # 50x / 50x / 30x token targets: text and image contribute 2016 packed positions
                     # each, audio 3024, so equal shares would not be equal codes. At the old 60/20/20
                     # the image expert was the binding constraint and the whole run had to be 1.83x
                     # larger to give it its tokens.
                     # Physics, chemistry and a second math-instruction source were carved OUT of
                     # their own modality rather than added on top: text still totals 0.4166 and
                     # image 0.4167, so the 41.7/41.7/16.7 sample budget - and with it the equal
                     # live-codes-per-modality argument above - is unchanged. Every pre-existing
                     # text source is scaled by 0.875, every pre-existing image source by 0.850.
                     mix=dict(text_fineweb=0.0728, text_cosmopedia=0.0425, text_wikipedia=0.0243,
                              text_math_web=0.0425, text_math_fine=0.0365, text_math_numina=0.0304,
                              text_arxiv=0.0365, text_code=0.0547, text_code_inst=0.0243,
                              text_physics=0.0208, text_chemistry=0.0208, text_math_inst=0.0104,
                              image_docvqa=0.0886, image_imagenet=0.0886, image_coco=0.0708,
                              image_wikiart=0.0531, image_flowers=0.0354, image_food=0.0177,
                              image_latex=0.0208, image_scienceqa=0.0417,
                              audio_libri=0.0625, audio_libritts=0.0417, audio_fleurs=0.0417,
                              audio_peoples=0.0208),
                     # "auto" = learned codec where this machine can build one (text: always,
                     # image: needs a k-means codebook file, audio: needs descript-audio-codec),
                     # otherwise the stand-in. The winner and the reason land in every shard's
                     # meta, so a run can never be mislabelled later. See docs/codecs.md.
                     codebook_size=256, prefer_real=True, codec_tier="auto",
                     K=None, R=None, base_len=None),
    )


def train_cfg(cfg, mode="strict", out_dir=None, steps=None):
    """The flat config one trainer reads: what write_cfgs puts on disk, built in memory.

    write_cfgs and the notebook both go through this, so a direct-feed run differs from a shard
    run in the data source and nothing else.
    """
    prod = cfg["produce"]
    t = dict(cfg["train"])
    if steps:
        t["steps"] = int(steps)
    t.update(sched=cfg["sched"], vocab=cfg["vocab"],
             model=dict(cfg["model"], mask_mode=mode),
             out_dir=out_dir or os.path.join(RUN, "train_" + mode),
             shard_dir=prod["out_dir"],
             producer_log=os.path.join(RUN, "producer.log"),
             eval_files={g: os.path.join(prod["out_dir"], f"eval_{g}.npz")
                         for g in prod["holdout"]},
             consume=False,
             stream_timeout=1800,
             preflight=cfg.get("preflight") or {})
    return t


def write_cfgs(cfg, budget=None, steps=None):
    from omni.schedule import OmniSchedule
    os.makedirs(os.path.join(RUN, "cfg"), exist_ok=True)
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
                         modalities=tuple(sch.get("modalities") or MODALITIES))
    prod = dict(cfg["produce"])
    if budget:
        prod["sample_budget"] = budget
    prod.update(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
                codebook_size=cfg["produce"]["codebook_size"])
    prod.setdefault("only_sources", None)
    paths = dict(
        produce=os.path.join(RUN, "cfg", "produce.json"),
        train_block=os.path.join(RUN, "cfg", "train_block.json"),
        train_ar=os.path.join(RUN, "cfg", "train_ar.json"),
        train_strict=os.path.join(RUN, "cfg", "train_strict.json"),
        eval=os.path.join(RUN, "cfg", "eval.json"),
    )
    json.dump(prod, open(paths["produce"], "w"), indent=2)

    for arm, mode in (("train_block", "block"), ("train_ar", "ar"),
                      ("train_strict", "strict")):
        t = train_cfg(cfg, mode, os.path.join(RUN, arm), steps)
        json.dump(t, open(paths[arm], "w"), indent=2)

    json.dump(dict(block_ckpt=os.path.join(RUN, "train_block", "final.pt"),
                   ar_ckpt=os.path.join(RUN, "train_ar", "final.pt"),
                   strict_ckpt=os.path.join(RUN, "train_strict", "final.pt"),
                   eval_files=sorted(glob.glob(os.path.join(prod["out_dir"], "eval_*.npz"))),
                   # 0 = every held-out row. 48 of the ~320 rows a x4 carve produces threw
                   # away 85% of the eval signal, and the AR-parity score reads one slot per
                   # modality per row, so its error bar is set by this number.
                   n_samples=0, out=os.path.join(RUN, "eval_table.json")),
              open(paths["eval"], "w"), indent=2)
    print("wrote", ", ".join(paths.values()))
    return paths


def cmd_check(args):
    import torch
    from omni import model as M
    from omni.data import collate, synthetic_batch
    from omni.schedule import OmniSchedule
    from omni.train import build, n_params

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    M.selfcheck()

    cfg = default_cfg()
    sch = cfg["sched"]
    m, sched = build(dict(sched=sch, vocab=cfg["vocab"], model=cfg["model"]))
    print(f"model: {n_params(m)/1e6:.2f}M params | packed {sched.total_len} positions/sample")
    print(sched.describe())

    b = synthetic_batch(sched, cfg["vocab"], B=3, present_all=False)
    out = m(b)
    print("logits:", {f"{k[0]}_L{k[1]}": tuple(v.shape) for k, v in list(out.items())[:4]}, "...")

    prod = subprocess.run([sys.executable, "-u", "-m", "omni.producer", "--selfcheck",
                           _write_tmp(prod_cfg_only(cfg))],
                          cwd=HERE, capture_output=True, text=True)
    print(prod.stdout[-800:] or prod.stderr[-800:])
    assert prod.returncode == 0, "producer selfcheck failed"
    print("PIPELINE CHECK OK")


def prod_cfg_only(cfg):
    from omni.schedule import OmniSchedule
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
                         modalities=tuple(sch.get("modalities") or MODALITIES))
    p = {k: v for k, v in cfg["produce"].items() if k not in ("K", "R", "base_len")}
    p.update(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
             codebook_size=cfg["produce"]["codebook_size"], only_sources=None)
    return p


def _write_tmp(obj):
    path = os.path.join(RUN, "cfg", "produce_selfcheck.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
    return path


def _nshards(d):
    try:
        return sum(1 for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".npz"))
    except FileNotFoundError:
        return 0


def cmd_produce(args):
    cfg = apply_overrides(default_cfg(), args)
    # ShardStream reads with consume=False, so shard files are never deleted: the rolling-buffer
    # cap (max_ahead_shards) can only ever block forever once it is hit. Batch production wants
    # the whole corpus on disk, so allow the cap to be raised explicitly.
    if getattr(args, "max_ahead", 0):
        cfg["produce"]["max_ahead_shards"] = int(args.max_ahead)
    paths = write_cfgs(cfg, budget=args.budget)
    os.makedirs(RUN, exist_ok=True)
    nworkers = max(1, args.workers)
    if nworkers == 1:
        log = open(os.path.join(RUN, "producer.log"), "w")
        p = subprocess.Popen([sys.executable, "-u", "-m", "omni.producer", paths["produce"]],
                             cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        print("producer pid", p.pid, "-> ", os.path.join(RUN, "producer.log"))
        if args.foreground:
            p.wait()
            return p.returncode
        print("background; training can start once EVAL_READY exists")
        return 0

    # ---- parallel: N processes, each reading a disjoint slice (index % N) of every stream.
    # The held-out blocks come from worker 0 only, and this process owns EVAL_READY/DONE, so a
    # half-finished fan-out can never look ready.
    base = json.load(open(paths["produce"]))
    stream_dir = base["out_dir"]                 # cfg from default_cfg() has no out_dir
    extra = sum(base["holdout"].values())        # worker 0's budget is spent on the eval blocks
    for f in ("EVAL_READY", "DONE", "STOP"):
        fp = os.path.join(stream_dir, f)
        if os.path.exists(fp):
            os.remove(fp)
    total = int(base["sample_budget"])
    per = total // nworkers
    procs, logs = [], []
    for i in range(nworkers):
        c = json.loads(json.dumps(base))
        c["sample_budget"] = per + (total - per * nworkers if i == 0 else 0) + (extra if i == 0 else 0)
        c["worker"] = [i, nworkers]
        cp = os.path.join(RUN, "cfg", f"produce_w{i}.json")
        json.dump(c, open(cp, "w"), indent=2)
        f = open(os.path.join(RUN, f"producer_w{i}.log"), "w")
        logs.append(f)
        if i:
            time.sleep(2.5)                  # spread the initial burst of stream opens
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "-m", "omni.producer", cp, "--worker", f"{i}/{nworkers}"],
            cwd=HERE, stdout=f, stderr=subprocess.STDOUT))
    print(f"{nworkers} producers | budget {total} -> {per}/worker | logs run/producer_w*.log")
    if not args.foreground:
        print("background; training can start once EVAL_READY exists")
        return 0

    # emit() fills a shuffle buffer before feeding the writer, so the first shard cannot exist
    # until shuffle_buf + shard_samples samples have been pulled PER WORKER. Reporting only the
    # shard count makes that interval indistinguishable from a hang.
    first_at = int(base.get("shuffle_buf", 512)) + int(base.get("shard_samples", 64))
    print(f"  first shard needs ~{first_at} samples/worker ({first_at * nworkers:,} total, "
          f"shuffle_buf + shard_samples): 0 shards before that is normal", flush=True)
    t0, codes = time.time(), []
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(10)
            done = _nshards(stream_dir)
            got = 0
            for i in range(nworkers):                 # the worker's own counter, not the file count
                try:
                    got += int(json.load(open(os.path.join(stream_dir, f"progress_w{i}.json")))[
                        "samples"])
                except Exception:                     # noqa: BLE001 - not written yet
                    pass
            el = max(1e-9, time.time() - t0)
            print(f"  {done} shards | {got:,}/{total:,} samples | "
                  f"{got / el:.1f}/s | {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        raise
    codes = [p.wait() for p in procs]
    for f in logs:
        f.close()
    if any(codes):
        print("producer failure codes:", codes)
        return 1

    stats = []
    for i in range(nworkers):
        sp = os.path.join(stream_dir, f"stats_w{i}.json")
        if os.path.exists(sp):
            stats.append(json.load(open(sp)))
    samples = sum(s.get("samples", 0) for s in stats)
    shards = _nshards(stream_dir)
    mins = (time.time() - t0) / 60
    open(os.path.join(stream_dir, "EVAL_READY"), "w").write("1")
    out = dict(samples=samples, shards=shards, workers=nworkers,
               elapsed_min=round(mins, 2), samples_per_s=round(samples / max(mins * 60, 1e-9), 2),
               per_worker=[s.get("samples", 0) for s in stats],
               per_source={k: sum(s.get("per_source", {}).get(k, 0) for s in stats)
                           for k in (stats[0].get("per_source", {}) if stats else {})})
    json.dump(out, open(os.path.join(stream_dir, "stats.json"), "w"), indent=2)
    open(os.path.join(stream_dir, "DONE"), "w").write("1")
    print("DONE", {k: out[k] for k in ("samples", "shards", "workers", "elapsed_min",
                                       "samples_per_s")})
    return 0


def cmd_train(args):
    cfg = apply_overrides(default_cfg(), args)
    paths = write_cfgs(cfg, budget=args.budget, steps=args.steps)
    which = paths.get(f"train_{args.arm}") or paths["train_block"]
    if getattr(args, "arms", None):
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in arms if a not in ("strict", "ar", "block")]
        if bad:
            raise SystemExit(f"unknown arms {bad}; pick from strict, ar, block")
        tcfg = json.load(open(which))
        live = dict(cfg["produce"])
        live["sched"], live["vocab"] = cfg["sched"], cfg["vocab"]
        live["holdout"] = cfg["produce"]["holdout"]
        # One doc per draw. `batch_docs=8` makes `Source.batch` block until it has 8 decodable
        # rows, and a slow modality (video downloads, audio decode) then stalls the whole stream
        # for minutes while every other source sits idle. Weights already control the mix.
        # With the threaded pump a slow source can no longer block the others, so pull in chunks:
        # one network round trip per 8 rows instead of per row.
        live["batch_docs"] = 8
        tcfg["arms"] = arms
        if getattr(args, "live", False):
            tcfg["live_cfg"] = live
        tcfg["out_dir"] = RUN              # duplex saves into run/train_{arm}/
        os.makedirs(os.path.join(RUN, "cfg"), exist_ok=True)
        which = os.path.join(RUN, "cfg", "train_duplex.json")
        json.dump(tcfg, open(which, "w"), indent=1)
        print(f"duplex: arms {arms} | live stream, no shards | out {RUN}")
    if getattr(args, "inproc", False):
        # In a notebook the subprocess buys nothing and costs a second interpreter: `pip` lands in
        # the kernel's python, the subprocess may resolve a different one, and the hard exit that
        # exists only to escape the subprocess would kill the kernel. Call the loop directly.
        from omni.train import run_duplex
        tcfg["hard_exit"] = False
        print("in-process: hard exit disabled, the cell keeps running after the last step")
        return run_duplex(tcfg, smoke=args.smoke or None)
    cmd = [sys.executable, "-u", "-m", "omni.train", which]
    if args.smoke:
        cmd += ["--smoke", str(args.smoke)]
    return subprocess.call(cmd, cwd=HERE)


def cmd_eval(args):
    cfg = apply_overrides(default_cfg(), args)
    paths = write_cfgs(cfg)
    return subprocess.call([sys.executable, "-u", "-m", "omni.eval", paths["eval"]], cwd=HERE)


def cmd_all(args):
    """Producer in the background, then wait for the held-out blocks, then both arms, then eval."""
    cfg = apply_overrides(default_cfg(), args)
    paths = write_cfgs(cfg, budget=args.budget, steps=args.steps)
    os.makedirs(RUN, exist_ok=True)
    stream_dir = cfg["produce"]["out_dir"]
    log = open(os.path.join(RUN, "producer.log"), "w")
    p = subprocess.Popen([sys.executable, "-u", "-m", "omni.producer", paths["produce"]],
                         cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    print("producer pid", p.pid)
    t0 = time.time()
    while not os.path.exists(os.path.join(stream_dir, "EVAL_READY")):
        if p.poll() is not None:
            print("producer exited early; tail:")
            print("".join(open(os.path.join(RUN, "producer.log")).readlines()[-25:]))
            return 1
        if args.wait_timeout and time.time() - t0 > args.wait_timeout:
            print(f"no held-out blocks after {args.wait_timeout}s; tail:")
            print("".join(open(os.path.join(RUN, "producer.log")).readlines()[-25:]))
            return 1
        time.sleep(5)
    print(f"EVAL_READY after {time.time() - t0:.0f}s")
    rc = 0
    for arm in ("block", "ar"):
        cmd = [sys.executable, "-u", "-m", "omni.train", paths[f"train_{arm}"]]
        if args.smoke:
            cmd += ["--smoke", str(args.smoke)]
        rc |= subprocess.call(cmd, cwd=HERE)
    if not args.smoke:
        rc |= subprocess.call([sys.executable, "-u", "-m", "omni.eval", paths["eval"]], cwd=HERE)
    open(os.path.join(stream_dir, "STOP"), "w").write("1")     # let a back-pressured producer exit
    p.terminate()
    return rc


def build_parser():
    """The CLI, exposed so a notebook can build the same cfg in-process: no cfg json on disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "produce", "train", "eval", "all", "cfg"])
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--arm", choices=["block", "ar", "strict"], default="block")
    ap.add_argument("--inproc", action="store_true",
                    help="run the duplex loop in this process instead of spawning a trainer; use "
                         "from a notebook so the kernel's python, packages and output are kept")
    ap.add_argument("--live", action="store_true",
                    help="feed the duplex run from the Hub directly instead of shards. Measured "
                         "1.5 samples/s in one process vs 45 for a sharded producer, so this is "
                         "for a short kill test where a corpus is unwanted, not for a real run.")
    ap.add_argument("--log-every", type=int, default=None, dest="log_every",
                    help="steps between metric lines. Default 10 = one line every ~100 s on a live "
                         "stream; pass 1 to get a line every step so silence always means a stall")
    ap.add_argument("--save-every", type=int, default=None, dest="save_every",
                    help="checkpoint interval; the live-stream path has no replay, so this is "
                         "the only way to survive a Colab session limit")
    ap.add_argument("--shuffle-buf", type=int, default=0, dest="shuffle_buf",
                    help="rows buffered per worker before the first shard can be written "
                         "(default 512; lower it for smoke corpora)")
    ap.add_argument("--eval-every", type=int, default=None, dest="eval_every",
                    help="validation loss every N steps on the held-out shards (default 1000)")
    ap.add_argument("--arms", default=None,
                    help="comma list to train in one process off the live stream, e.g. strict,ar. "
                         "No shard corpus is produced or read: both arms step the same batch.")
    ap.add_argument("--foreground", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="producer processes; each reads index i mod N of every stream")
    ap.add_argument("--moe", type=int, default=0,
                    help="n_experts (1 per modality). 0 = dense trunk, the old layout")
    ap.add_argument("--moe-ffn", type=int, default=0, dest="moe_ffn",
                    help="expert SwiGLU width h. 0 = mlp_ratio 4.0 (2048 at d=768). 4880 gives "
                         "359.8M per expert at d=1536/L=16, i.e. the 360M rung")
    ap.add_argument("--hetero-vocab", type=int, default=0, dest="hetero_vocab",
                    help="finest-level text vocab, e.g. 8000 or 32000; 0 = one codebook per level")
    ap.add_argument("--preflight-lr", type=float, default=None, dest="preflight_lr",
                    help="preflight learning rate. Default 2x the preset's train lr, clamped to "
                         "[3e-4, 3e-3] - a fixed 3e-3 is 15x too hot for md")
    ap.add_argument("--preflight-steps", type=int, default=None, dest="preflight_steps",
                    help="test C steps that run before training (default 300)")
    ap.add_argument("--no-preflight", action="store_true", dest="no_preflight",
                    help="skip test C - only to debug the trainer itself")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="sm=10.44M (debug), md=84.6M (kill test), lg=454M")
    ap.add_argument("--base-len", type=int, default=0,
                    help="text/image finest length; audio scales x1.5, video stays 32; "
                         "0 keeps 64 (504 packed positions). 1024 => ~7.1k positions")
    ap.add_argument("--r", default=None,
                    help="compression ratio: one int for every modality, or "
                         "'text=2,image=4,audio=4'. Level k = base_len / R**(K-1-k), so a "
                         "modality whose level 0 is one code for the whole sample (K=5,R=4 "
                         "text) has nothing learnable there. `check` prints the ladder and "
                         "warns when that happens.")
    ap.add_argument("--holdout-mult", type=int, default=0,
                    help="multiply the held-out block sizes (4 => 128/96/64/32, SE ~0.028)")
    ap.add_argument("--batch", type=int, default=0,
                    help="micro-batch. This is the dominant throughput knob: the step is "
                         "launch-bound long before it is FLOP-bound, so 8 -> 32-64 buys most of "
                         "the missing TFLOPS. Raise lr ~sqrt(batch/8) when you do.")
    ap.add_argument("--no-compile", action="store_true", dest="no_compile",
                    help="train eager instead of torch.compile(model). Measured on an 8 GB "
                         "RTX 3070 laptop it is 1.7x slower per step (186 vs 323 ms at "
                         "batch 16) and needs more VRAM; compile costs ~8-75 s at startup")
    ap.add_argument("--max-ahead", type=int, default=0,
                    help="producer rolling-buffer cap in shards; 0 keeps the default 24, "
                         "which blocks forever when nothing deletes shards (batch runs: 100000)")
    ap.add_argument("--wait-timeout", type=int, default=3600)
    return ap


def main(argv=None):
    """argv=None keeps the CLI; a notebook passes a list and never leaves the kernel."""
    a = build_parser().parse_args(argv)
    if a.cmd == "cfg":
        write_cfgs(apply_overrides(default_cfg(), a), a.budget)
        return 0
    return {"check": cmd_check, "produce": cmd_produce, "train": cmd_train,
            "eval": cmd_eval, "all": cmd_all}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
