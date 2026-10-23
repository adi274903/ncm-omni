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
#   md  84.6M   d768  L13 h12 kv3     -> the kill test rung
#   lg  454.1M  d1536 L18 h16 kv4     -> the paper number, rented H100 only
PRESETS = {
    "sm": dict(n_embd=384,  n_layer=6,  n_head=6,  n_kv_head=2, batch=8,  lr=3e-4,   warmup=100),
    "md": dict(n_embd=768,  n_layer=13, n_head=12, n_kv_head=3, batch=8,  lr=2e-4,   warmup=200),
    "lg": dict(n_embd=1536, n_layer=18, n_head=16, n_kv_head=4, batch=8,  lr=1.5e-4, warmup=300),
}


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
    n = getattr(a, "base_len", 0)
    if n:
        r = n / 64.0                       # 64 is the reference text/image length
        cfg["sched"]["base_len"] = dict(text=int(n), image=int(n),
                                        audio=max(4, int(round(96 * r / 4)) * 4),
                                        video=32)
        # video stays at 32: its base_len is frames x grid^2, so 512 would read 128 frames of
        # every clip. It is data-path only and currently dead in eval, so it is not worth the
        # decode. Raise it in build_codec if that ever changes.
    m = getattr(a, "holdout_mult", 0)
    if m:
        cfg["produce"]["holdout"] = {k: int(v * m) for k, v in cfg["produce"]["holdout"].items()}
    return cfg


def write_cfgs_all(cfg, budget=None, steps=None):
    return write_cfgs(cfg, budget=budget, steps=steps)


def default_cfg(budget=20000, out=os.path.join(RUN, "stream")):
    return dict(
        sched=dict(base_len=dict(text=64, image=64, audio=96, video=32), K=6, R=2, null_code=0),
        vocab=dict(text=257, image=257, audio=257, video=257),   # codebook_size + the NULL slot
        model=dict(n_embd=384, n_layer=6, n_head=6, n_kv_head=2, dropout=0.0,
                   qk_norm=True, mask_mode="block"),
        train=dict(steps=2000, batch=8, lr=3e-4, warmup=100, wd=0.01, clip=1.0,
                   log_every=10, eval_every=250, save_every=500),
        produce=dict(out_dir=out, sample_budget=budget, shard_samples=64,
                     max_ahead_shards=24, shuffle_buf=512, batch_docs=8, seed=3407,
                     holdout=dict(text=32, image=24, audio=16, video=8),
                     mix=dict(text_fineweb=0.14, text_cosmopedia=0.08, text_wikipedia=0.05,
                              text_math_web=0.08, text_math_fine=0.07, text_math_numina=0.06,
                              text_arxiv=0.07, text_code=0.10, text_code_inst=0.05,
                              image_docvqa=0.05, image_imagenet=0.05, image_coco=0.04,
                              image_wikiart=0.03, image_flowers=0.02, image_food=0.01,
                              audio_libri=0.03, audio_libritts=0.02, audio_fleurs=0.02,
                              audio_peoples=0.01,
                              video_ucf=0.02),
                     codebook_size=256, prefer_real=True,
                     K=None, R=None, base_len=None),
    )


def write_cfgs(cfg, budget=None, steps=None):
    from omni.schedule import OmniSchedule
    os.makedirs(os.path.join(RUN, "cfg"), exist_ok=True)
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"])
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
        t = dict(cfg["train"])
        if steps:
            t["steps"] = int(steps)
        t.update(sched=sch, vocab=cfg["vocab"],
                 model=dict(cfg["model"], mask_mode=mode),
                 out_dir=os.path.join(RUN, arm),
                 shard_dir=prod["out_dir"],
                 producer_log=os.path.join(RUN, "producer.log"),
                 eval_files={g: os.path.join(prod["out_dir"], f"eval_{g}.npz")
                             for g in prod["holdout"]},
                 consume=False,          # both arms read the same shards
                 stream_timeout=1800)
        json.dump(t, open(paths[arm], "w"), indent=2)

    import glob
    json.dump(dict(block_ckpt=os.path.join(RUN, "train_block", "final.pt"),
                   ar_ckpt=os.path.join(RUN, "train_ar", "final.pt"),
                   strict_ckpt=os.path.join(RUN, "train_strict", "final.pt"),
                   eval_files=sorted(glob.glob(os.path.join(prod["out_dir"], "eval_*.npz"))),
                   n_samples=48, out=os.path.join(RUN, "eval_table.json")),
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

    prod = subprocess.run([sys.executable, "-m", "omni.producer", "--selfcheck",
                           _write_tmp(prod_cfg_only(cfg))],
                          cwd=HERE, capture_output=True, text=True)
    print(prod.stdout[-800:] or prod.stderr[-800:])
    assert prod.returncode == 0, "producer selfcheck failed"
    print("PIPELINE CHECK OK")


def prod_cfg_only(cfg):
    from omni.schedule import OmniSchedule
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"])
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
        p = subprocess.Popen([sys.executable, "-m", "omni.producer", paths["produce"]],
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
            [sys.executable, "-m", "omni.producer", cp, "--worker", f"{i}/{nworkers}"],
            cwd=HERE, stdout=f, stderr=subprocess.STDOUT))
    print(f"{nworkers} producers | budget {total} -> {per}/worker | logs run/producer_w*.log")
    if not args.foreground:
        print("background; training can start once EVAL_READY exists")
        return 0

    t0, codes = time.time(), []
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(10)
            done = _nshards(stream_dir)
            print(f"  {done} shards on disk | {time.time() - t0:.0f}s", flush=True)
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
    cmd = [sys.executable, "-m", "omni.train", which]
    if args.smoke:
        cmd += ["--smoke", str(args.smoke)]
    return subprocess.call(cmd, cwd=HERE)


def cmd_eval(args):
    cfg = apply_overrides(default_cfg(), args)
    paths = write_cfgs(cfg)
    return subprocess.call([sys.executable, "-m", "omni.eval", paths["eval"]], cwd=HERE)


def cmd_all(args):
    """Producer in the background, then wait for the held-out blocks, then both arms, then eval."""
    cfg = apply_overrides(default_cfg(), args)
    paths = write_cfgs(cfg, budget=args.budget, steps=args.steps)
    os.makedirs(RUN, exist_ok=True)
    stream_dir = cfg["produce"]["out_dir"]
    log = open(os.path.join(RUN, "producer.log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "omni.producer", paths["produce"]],
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
        cmd = [sys.executable, "-m", "omni.train", paths[f"train_{arm}"]]
        if args.smoke:
            cmd += ["--smoke", str(args.smoke)]
        rc |= subprocess.call(cmd, cwd=HERE)
    if not args.smoke:
        rc |= subprocess.call([sys.executable, "-m", "omni.eval", paths["eval"]], cwd=HERE)
    open(os.path.join(stream_dir, "STOP"), "w").write("1")     # let a back-pressured producer exit
    p.terminate()
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "produce", "train", "eval", "all", "cfg"])
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--arm", choices=["block", "ar", "strict"], default="block")
    ap.add_argument("--foreground", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="producer processes; each reads index i mod N of every stream")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="sm=10.44M (debug), md=84.6M (kill test), lg=454M")
    ap.add_argument("--base-len", type=int, default=0,
                    help="text/image finest length; audio scales x1.5, video stays 32; "
                         "0 keeps 64 (504 packed positions). 1024 => ~7.1k positions")
    ap.add_argument("--holdout-mult", type=int, default=0,
                    help="multiply the held-out block sizes (4 => 128/96/64/32, SE ~0.028)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--max-ahead", type=int, default=0,
                    help="producer rolling-buffer cap in shards; 0 keeps the default 24, "
                         "which blocks forever when nothing deletes shards (batch runs: 100000)")
    ap.add_argument("--wait-timeout", type=int, default=3600)
    a = ap.parse_args()
    if a.cmd == "cfg":
        write_cfgs(apply_overrides(default_cfg(), a), a.budget)
        return 0
    return {"check": cmd_check, "produce": cmd_produce, "train": cmd_train,
            "eval": cmd_eval, "all": cmd_all}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
