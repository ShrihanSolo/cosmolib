"""
weekly_summary.py

One run on Midway that collects everything the weekly-update slides need about the chains:

  1. job status   : for each of the 22 LSST rerun jobs, whether its log says sampling finished, and whether
                    the chain file exists (a "FAILED" sacct state after "Sampling posterior: done" is fine)
  2. shift table  : for every imprinted chain, the shift relative to its fiducial chain in fiducial sigma
                    (S8 and Omega_m posterior means, like Table2), for the NEW chains and for the OLD ones
                    (LSST: old_results_8 = before the row-order fix; DES/DECADE: old_results_7 = before the
                    12 Nov reruns), so the slides can show how much each fix changed the result
  3. truth offset : each fiducial chain's S8 and Omega_m mean minus the sim truth, in its own sigma
  4. samples      : 10k equal-weight (Omega_m, S8) draws per chain (npz) so the slide figures can be
                    drawn off Midway; plus quick-look contour PNGs per survey

Run from cosmolib/shrihan, in the cosmosis3 environment (numpy, matplotlib; scipy/getdist not needed):
    python weekly_summary.py
Writes everything into cosmolib/shrihan/weekly_2026-11-17/ (about 4 MB). To send it back:
    cd ~/weaklens/cosmolib && git add -f shrihan/weekly_2026-11-17 shrihan/weekly_summary.py \
        && git commit -m "weekly summary outputs" && git push
It only reads chains and logs; it never touches the running jobs. Safe to rerun at any time.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                     # cosmolib/shrihan
COSMOLIB = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str((COSMOLIB / "output").resolve()), help="chain directory")
ap.add_argument("--batch", default="/project/chihway/shrihan/clib_batch_files", help="batch files and slurm logs")
ap.add_argument("--out", default=str(HERE / "weekly_2026-11-17"), help="output directory")
ap.add_argument("--no-plots", action="store_true")
args = ap.parse_args()
RUNS, BATCH, OUT = Path(args.runs), Path(args.batch), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

TRUTH = {"S8": 0.7959, "Om": 0.3013}                        # CAMB 1.4.0 for the sim cosmology (2026-11-15)
COL_OM, COL_S8, COL_LOGW = "cosmological_parameters--omega_m", "COSMOLOGICAL_PARAMETERS--SIGMA_8", "log_weight"
NON_PARAM = {"log_weight", "prior", "post", "like", "weight"}
COSMO = "cosmological_parameters"
rng = np.random.default_rng(17)

# survey -> (chain stem, folder with the chains from before this week's fix, imprint tags to compare)
SURVEYS = {
    "DES Y3": ("shrihan-clean-des-y3", "old_results_7", ["psfA", "respA", "dcrA", "csA"]),
    "DECADE": ("shrihan-clean-delve-dr3", "old_results_7", ["psfA", "respA", "dcrA", "csA"]),
    "LSST Y1": ("shrihan-clean-lsst-y1", "old_results_8", ["psfA", "respA", "dcrA", "csA", "dcrA3x", "dcrA9x"]),
    "LSST Y5": ("shrihan-clean-lsst-y5", "old_results_8", ["psfA", "respA", "dcrA", "csA"]),
    "LSST Y10": ("shrihan-clean-lsst-y10", "old_results_8",
                 ["psfA", "psfArn", "respA", "dcrA", "csA", "respA3x", "csA3x", "dcrA3x", "dcrA9x",
                  "psfArn3x", "psfArn9x"]),
}
# Y10 DCR x9 is new (it replaced x7), so its "old" comparison is the old x7 chain
OLD_ALIAS = {"shrihan-clean-lsst-y10-dcrA9x": "shrihan-clean-lsst-y10-dcrA7x"}
# Y10 fid/dcr were not rerun: the current files are the restored *fixed* chains; the old ones are the
# misordered originals, which the archive keeps under the plain names


def load_chain(path):
    """Columns (sampled params + S8) and normalised weights; None if the file has no samples yet."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0 or data.shape[1] != len(header):
        return None
    cols = {n: data[:, i] for i, n in enumerate(header)}
    sampled = [n for n in header if n == n.lower() and n not in NON_PARAM]
    ok = np.all([np.isfinite(cols[n]) for n in sampled + [COL_S8, COL_LOGW]], axis=0)
    w = np.exp(cols[COL_LOGW][ok] - np.nanmax(cols[COL_LOGW][ok]))
    w /= w.sum()
    p = {n: cols[n][ok] for n in sampled}
    p["S8"] = cols[COL_S8][ok] * np.sqrt(cols[COL_OM][ok] / 0.3)
    p["Om"] = cols[COL_OM][ok]
    return {"p": p, "w": w, "ess": float(1 / np.sum(w ** 2)), "mtime": path.stat().st_mtime}


def ms(c, n):
    m = float(np.sum(c["w"] * c["p"][n]))
    return m, float(np.sqrt(np.sum(c["w"] * (c["p"][n] - m) ** 2)))


def shifts(fid, sys_):
    """Shift of every shared parameter, (mean_sys - mean_fid) / sigma_fid."""
    out = {}
    for n in fid["p"]:
        if n in sys_["p"] and n != COL_OM:
            mf, sf = ms(fid, n)
            out[n] = (ms(sys_, n)[0] - mf) / sf if sf > 0 else float("nan")
    return out


def short(n):
    return n.split("--")[-1]


chain_cache = {}


def chain(path):
    path = Path(path)
    if path not in chain_cache:
        chain_cache[path] = load_chain(path)
    return chain_cache[path]


# ---------------------------------------------------------------- 1. job status
print("\n1. LSST rerun jobs (log check)")
jobs = []
for b in sorted(BATCH.glob("cosmorun_LSSTY*.batch")):
    text = b.read_text()
    job = re.search(r"--job-name[ =](\S+)", text)
    ini = re.findall(r"cosmosis --mpi (\S+)", text)
    if not job or not ini:
        continue
    job = job.group(1)
    out = None
    inip = COSMOLIB / ini[0]
    if inip.exists():
        for line in open(inip):
            if line.startswith("filename = output"):
                out = RUNS / Path(line.split("=", 1)[1].strip()).name   # ini writes output/<name>
    log = BATCH / f"slurm-{job}.out"
    logtxt = log.read_text(errors="replace") if log.exists() else ""
    done = ("Sampling posterior: done" in logtxt) or ("Total posterior evaluations" in logtxt)
    c = chain(out) if out else None
    jobs.append({"job": job, "batch": b.name, "chain": out.name if out else None, "log": log.exists(),
                 "sampling_done": done, "chain_rows": int(len(c["w"])) if c else 0,
                 "ess": round(c["ess"]) if c else 0})
# sacct states, if available (Midway3 sacct works for the user's own jobs)
try:
    r = subprocess.run(["sacct", "-u", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip(),
                        "-S", "2026-11-15", "-X", "-n", "-P", "--format=JobName,JobID,State,Elapsed,ExitCode"],
                       capture_output=True, text=True, timeout=60)
    sacct = {}
    for line in r.stdout.splitlines():
        name, jid, state, el, ex = (line.split("|") + [""] * 5)[:5]
        sacct[name] = {"id": jid, "state": state, "elapsed": el, "exit": ex}   # latest submission wins
except Exception as e:  # noqa: BLE001
    sacct = {}
    print(f"  (sacct not available: {e})")
for j in jobs:
    s = sacct.get(j["job"], {})
    j.update({"slurm_id": s.get("id"), "state": s.get("state"), "elapsed": s.get("elapsed"), "exit": s.get("exit")})
    verdict = ("finished" if j["sampling_done"] and j["chain_rows"] else
               "running/pending" if (s.get("state") or "").startswith(("RUNNING", "PENDING")) else
               "CHECK" if j["sampling_done"] else "not finished")
    j["verdict"] = verdict
    print(f"  {j['job']:22s} {str(j['state']):12s} {str(j['elapsed']):12s} chain rows {j['chain_rows']:6d}  -> {verdict}")

# ---------------------------------------------------------------- 2-3. shifts and truth offsets
print("\n2. shifts of imprinted chains relative to their fiducial, in fiducial sigma (new vs old)")
rows, fids, samples = [], [], {}


def keep_samples(key, c):
    if c is not None:
        idx = rng.choice(len(c["w"]), size=10000, p=c["w"])
        samples[key] = np.column_stack([c["p"]["Om"][idx], c["p"]["S8"][idx]]).astype(np.float32)


for survey, (stem, oldname, tags) in SURVEYS.items():
    olddir = RUNS / oldname
    fn = RUNS / f"{stem}-fidA-nautilus.txt"
    fo = olddir / f"{stem}-fidA-nautilus.txt"
    fnew, fold = chain(fn), chain(fo)
    if fold is None:          # DES/DECADE fiducials were not rerun: old = new
        fold = fnew
    keep_samples(f"{stem}-fidA|new", fnew)
    keep_samples(f"{stem}-fidA|old", fold)
    for tag_, f in (("new", fnew), ("old", fold)):
        if f:
            rec = {"survey": survey, "which": tag_, "ess": round(f["ess"])}
            for n in ("S8", "Om"):
                m, s = ms(f, n)
                rec.update({f"{n}_mean": m, f"{n}_sigma": s, f"{n}_vs_truth_sigma": (m - TRUTH[n]) / s})
            fids.append(rec)
    print(f"===== {survey} =====  fiducial new: {'ok' if fnew else 'MISSING'}  old: {'ok' if fold else 'MISSING'}")
    for t in tags:
        name = f"{stem}-{t}"
        cn = chain(RUNS / f"{name}-nautilus.txt")
        oldp = olddir / f"{OLD_ALIAS.get(name, name)}-nautilus.txt"
        co = chain(oldp)
        keep_samples(f"{name}|new", cn)
        keep_samples(f"{name}|old", co)
        row = {"survey": survey, "chain": t, "old_is": OLD_ALIAS.get(name, name).split("-")[-1]}
        for lab, c, f in (("new", cn, fnew), ("old", co, fold)):
            if c is None or f is None:
                row[lab] = None
                continue
            sh = shifts(f, c)
            nuis = sorted(((abs(v), short(k), v) for k, v in sh.items()
                           if not k.startswith(COSMO + "--") and k not in ("S8", "Om") and np.isfinite(v)),
                          reverse=True)[:3]
            row[lab] = {"dS8": sh["S8"], "dOm": sh["Om"], "ess": round(c["ess"]),
                        "nuisance_top": [(n, round(v, 3)) for _, n, v in nuis]}
        rows.append(row)

        def f2(d, k):
            return f"{d[k]:+.2f}" if d else "  -- "
        dd = (row["new"]["dS8"] - row["old"]["dS8"]) if row["new"] and row["old"] else float("nan")
        print(f"  {t:9s} dS8 new {f2(row['new'], 'dS8')} old {f2(row['old'], 'dS8')} (change {dd:+.2f})   "
              f"dOm new {f2(row['new'], 'dOm')} old {f2(row['old'], 'dOm')}")

print("\n3. fiducial chains against the sim truth (S8 0.7959, Om 0.3013), in each chain's own sigma")
for r in fids:
    print(f"  {r['survey']:9s} {r['which']}: S8 {r['S8_mean']:.4f} +- {r['S8_sigma']:.4f} ({r['S8_vs_truth_sigma']:+.2f} sigma)"
          f"   Om {r['Om_mean']:.4f} +- {r['Om_sigma']:.4f} ({r['Om_vs_truth_sigma']:+.2f} sigma)")

summary = {"truth": TRUTH, "jobs": jobs, "shifts": rows, "fiducials": fids, "runs_dir": str(RUNS)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
np.savez_compressed(OUT / "samples_om_s8.npz", **samples)

# ---------------------------------------------------------------- 4. quick-look plots
if not args.no_plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def contour(ax, xy, color, ls="-", label=None):
        H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=60)
        # light smoothing so the 68/95% levels are not jagged
        k = np.array([1, 4, 6, 4, 1.0]); k /= k.sum()
        H = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, H)
        H = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, H)
        s = np.sort(H.ravel())[::-1]; c = np.cumsum(s) / s.sum()
        lev = sorted({s[np.searchsorted(c, 0.95)], s[np.searchsorted(c, 0.68)]})
        X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")
        cs = ax.contour(X, Y, H, levels=lev, colors=[color], linestyles=ls, linewidths=1.3)
        if label:
            ax.plot([], [], color=color, ls=ls, label=label)
        return cs

    for survey, (stem, _, tags) in SURVEYS.items():
        if not any(k.startswith(stem + "-") for k in samples):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
        for ax, which in zip(axes, ("new", "old")):
            cols = plt.cm.tab10.colors
            if f"{stem}-fidA|{which}" in samples:
                contour(ax, samples[f"{stem}-fidA|{which}"], "k", label="fiducial")
            for i, t in enumerate(tags):
                key = f"{stem}-{t}|{which}"
                if key in samples:
                    contour(ax, samples[key], cols[i % 10], label=t)
            ax.axvline(TRUTH["Om"], color="0.6", lw=0.8); ax.axhline(TRUTH["S8"], color="0.6", lw=0.8)
            ax.set_title(f"{survey}: {which} chains"); ax.set_xlabel(r"$\Omega_m$")
            ax.legend(fontsize=7, loc="upper right")
        axes[0].set_ylabel(r"$S_8$")
        fig.tight_layout()
        fig.savefig(OUT / f"quick_{stem.split('clean-')[1]}.png", dpi=110)
        plt.close(fig)

print(f"\nWrote {OUT}/summary.json, samples_om_s8.npz and quick_*.png.")
n_fin = sum(j["verdict"] == "finished" for j in jobs)
print(f"Conclusion: {n_fin} of {len(jobs)} LSST jobs have finished chains; "
      f"{sum(r['new'] is not None for r in rows)} of {len(rows)} imprint comparisons have a new chain. "
      "Send the folder back with the git commands in this file's header.")
