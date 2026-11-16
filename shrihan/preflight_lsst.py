"""
preflight_lsst.py

Checks everything the 22 LSST rerun jobs depend on before they are submitted (2026-11-15 row-order rerun):

  1. data vectors   : all 24 LSST 2pt files exist, are in CosmoCov row order, and were rebuilt after the fix
  2. inis           : each job's ini reads the right data vector and writes the expected chain name
  3. batch files    : same SLURM settings in all 22, 18 h limit, unique job names, log = slurm-<job>.out,
                      ini path exists
  4. chain directory: no old chain or nautilus checkpoint left that a job would resume from (resume = T);
                      the two restored Y10 chains (fidA, dcrA) are present and identical to the *fixed* runs

Run from cosmolib/shrihan, in the cosmosis3 environment (needs numpy + astropy):
    python preflight_lsst.py
"""
import argparse
import filecmp
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent                     # cosmolib/shrihan
COSMOLIB = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--batch", default="/project/chihway/shrihan/clib_batch_files", help="batch-file directory")
ap.add_argument("--runs", default=str((COSMOLIB / "output").resolve()), help="chain directory (output/ -> clib_runs)")
args = ap.parse_args()
BATCH = Path(args.batch)
RUNS = Path(args.runs)
OLD = RUNS / "old_results_8"
FIX_COMMIT = "9542ec7"                                      # first commit with the row-order fix
TIME = "18:00:00"

# batch file -> (ini, chain stem) for the 22 jobs. Y10 fidA/dcrA are not rerun (restored *fixed* chains).
JOBS = {
    "cosmorun_LSSTY1_fiducial_A.batch": ("lsst-y1-clean-fidA.ini", "shrihan-clean-lsst-y1-fidA"),
    "cosmorun_LSSTY1_psfimprint_A.batch": ("lsst-y1-clean-psfA.ini", "shrihan-clean-lsst-y1-psfA"),
    "cosmorun_LSSTY1_respimprint_A.batch": ("lsst-y1-clean-respA.ini", "shrihan-clean-lsst-y1-respA"),
    "cosmorun_LSSTY1_dcrimprint_A.batch": ("lsst-y1-clean-dcrA.ini", "shrihan-clean-lsst-y1-dcrA"),
    "cosmorun_LSSTY1_csimprint_A.batch": ("lsst-y1-clean-csA.ini", "shrihan-clean-lsst-y1-csA"),
    "cosmorun_LSSTY1_dcrimprint3x_A.batch": ("lsst-y1-clean-dcrA3x.ini", "shrihan-clean-lsst-y1-dcrA3x"),
    "cosmorun_LSSTY1_dcrimprint9x_A.batch": ("lsst-y1-clean-dcrA9x.ini", "shrihan-clean-lsst-y1-dcrA9x"),
    "cosmorun_LSSTY5_fiducial_A.batch": ("lsst-y5-clean-fidA.ini", "shrihan-clean-lsst-y5-fidA"),
    "cosmorun_LSSTY5_psfimprint_A.batch": ("lsst-y5-clean-psfA.ini", "shrihan-clean-lsst-y5-psfA"),
    "cosmorun_LSSTY5_respimprint_A.batch": ("lsst-y5-clean-respA.ini", "shrihan-clean-lsst-y5-respA"),
    "cosmorun_LSSTY5_dcrimprint_A.batch": ("lsst-y5-clean-dcrA.ini", "shrihan-clean-lsst-y5-dcrA"),
    "cosmorun_LSSTY5_csimprint_A.batch": ("lsst-y5-clean-csA.ini", "shrihan-clean-lsst-y5-csA"),
    "cosmorun_LSSTY10_psfimprint_A.batch": ("lsst-y10-clean-psfA.ini", "shrihan-clean-lsst-y10-psfA"),
    "cosmorun_LSSTY10_respimprint_A.batch": ("lsst-y10-clean-respA.ini", "shrihan-clean-lsst-y10-respA"),
    "cosmorun_LSSTY10_csimprint_A.batch": ("lsst-y10-clean-csA.ini", "shrihan-clean-lsst-y10-csA"),
    "cosmorun_LSSTY10_respimprint3x_A.batch": ("lsst-y10-clean-respA3x.ini", "shrihan-clean-lsst-y10-respA3x"),
    "cosmorun_LSSTY10_csimprint3x_A.batch": ("lsst-y10-clean-csA3x.ini", "shrihan-clean-lsst-y10-csA3x"),
    "cosmorun_LSSTY10_dcrimprint3x_A.batch": ("lsst-y10-clean-dcrA3x.ini", "shrihan-clean-lsst-y10-dcrA3x"),
    "cosmorun_LSSTY10_dcrimprint9x_A.batch": ("lsst-y10-clean-dcrA9x.ini", "shrihan-clean-lsst-y10-dcrA9x"),
    "cosmorun_LSSTY10_psfimprint_A_rn.batch": ("lsst-y10-clean-psfArn.ini", "shrihan-clean-lsst-y10-psfArn"),
    "cosmorun_LSSTY10_psfimprint_A_rn3x.batch": ("lsst-y10-clean-psfArn3x.ini", "shrihan-clean-lsst-y10-psfArn3x"),
    "cosmorun_LSSTY10_psfimprint_A_rn9x.batch": ("lsst-y10-clean-psfArn9x.ini", "shrihan-clean-lsst-y10-psfArn9x"),
}
RESTORED = ["shrihan-clean-lsst-y10-fidA", "shrihan-clean-lsst-y10-dcrA"]

fails = []


def report(ok, what, why=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + ("" if ok else f"  <- {why}"))
    if not ok:
        fails.append(f"{what}: {why}")


def ini_field(path, prefix):
    for line in open(path):
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------- 0. code version
print("\n0. code version")
has_fix = subprocess.run(["git", "-C", str(COSMOLIB), "merge-base", "--is-ancestor", FIX_COMMIT, "HEAD"]).returncode == 0
report(has_fix, f"cosmolib contains the row-order fix ({FIX_COMMIT})", "git pull cosmolib")

# ---------------------------------------------------------------- 1. data vectors
print("\n1. data vectors (2pt fits)")
sys.path.insert(0, str(HERE))
from make_lsst_2pt import VARIANTS  # noqa: E402
cov_order = {}
for y, name in (("LSSTY1", "lssty1"), ("LSSTY5", "lssty5"), ("LSSTY10", "lssty10")):
    c = np.loadtxt(HERE / f"covdata_{name}.txt", usecols=(0, 4, 5))
    m = {}
    for i, b1, b2 in c:
        m.setdefault(int(i), (int(b1) + 1, int(b2) + 1))
    cov_order[y] = m
fix_time = float(subprocess.run(["git", "-C", str(COSMOLIB), "log", "-1", "--format=%ct", FIX_COMMIT],
                                capture_output=True, text=True).stdout.strip() or 0)
for year, kind, scaling, name in VARIANTS:
    f = HERE / name / f"{name}.fits"
    if not f.exists():
        report(False, name, "missing: run make_lsst_2pt.py")
        continue
    with fits.open(f) as h:
        n = len(h["xip"].data)
        bad = sum(cov_order[year][off + i] != (int(d["BIN1"][i]), int(d["BIN2"][i])) or int(d["ANGBIN"][i]) != i % 20
                  for off, d in ((0, h["xip"].data), (n, h["xim"].data)) for i in range(n))
        cov_ok = h["COVMAT"].data.shape == (2 * n, 2 * n)
    fresh = f.stat().st_mtime > fix_time
    report(bad == 0 and cov_ok and fresh, name,
           f"{bad} rows out of order" if bad else ("covariance shape" if not cov_ok else "file older than the fix: rebuild"))

# ---------------------------------------------------------------- 2. inis
print("\n2. inis")
for batch, (ini, chain) in JOBS.items():
    p = HERE / "ini_files" / ini
    if not p.exists():
        report(False, ini, "missing")
        continue
    two = ini_field(p, "2PT_FILE")
    out = ini_field(p, "filename = output")
    two_ok = two is not None and (COSMOLIB / two).exists()
    out_ok = out == f"output/{chain}-nautilus.txt"
    report(two_ok and out_ok, f"{ini:30s} -> {chain}",
           "2PT_FILE missing" if not two_ok else f"writes {out}")

# ---------------------------------------------------------------- 3. batch files
print("\n3. batch files")
ref = None
names = {}
for batch, (ini, chain) in JOBS.items():
    p = BATCH / batch
    if not p.exists():
        report(False, batch, "missing")
        continue
    text = p.read_text()
    job = re.search(r"--job-name[ =](\S+)", text)
    job = job.group(1) if job else None
    outlog = re.search(r"--output=(\S+)", text)
    outlog = outlog.group(1) if outlog else None
    ini_used = re.findall(r"cosmosis --mpi (\S+)", text)
    ntasks = re.search(r"--ntasks=(\d+)", text)
    mpin = re.search(r"mpirun -n (\d+)", text)
    # Everything except the job-specific lines must be identical across the 22 files.
    common = "\n".join(l for l in text.splitlines()
                       if not re.search(r"--job-name|--output=|cosmosis --mpi", l) and l.strip() and l.strip() != "exit")
    if ref is None:
        ref, ref_name = common, batch
    problems = []
    if common != ref:
        problems.append(f"settings differ from {ref_name}")
    if f"--time={TIME}" not in text:
        problems.append(f"time limit is not {TIME}")
    if outlog != f"slurm-{job}.out":
        problems.append(f"log {outlog} != slurm-{job}.out")
    if ini_used != [f"shrihan/ini_files/{ini}"]:
        problems.append(f"runs {ini_used}")
    if not (ntasks and mpin and ntasks.group(1) == mpin.group(1)):
        problems.append("mpirun -n != --ntasks")
    if job in names:
        problems.append(f"job name {job} also used by {names[job]}")
    names[job] = batch
    report(not problems, f"{batch:42s} job {job}", "; ".join(problems))

# ---------------------------------------------------------------- 4. chain directory
print(f"\n4. chain directory {RUNS}")
for batch, (ini, chain) in JOBS.items():
    left = sorted(p.name for p in RUNS.glob(f"{chain}-nautilus.*"))
    report(not left, f"no old files for {chain}", f"still there: {left} (resume = T would continue them)")
for chain in RESTORED:
    txt = RUNS / f"{chain}-nautilus.txt"
    src = next((p for p in (RUNS / f"{chain.replace('A', 'Afixed', 1)}-nautilus.txt",
                            OLD / f"{chain.replace('A', 'Afixed', 1)}-nautilus.txt") if p.exists()), None)
    ckpt = sorted(p.name for p in RUNS.glob(f"{chain}-nautilus.sampler_status*"))
    ok = txt.exists() and src is not None and filecmp.cmp(txt, src, shallow=False) and not ckpt
    why = ("missing" if not txt.exists() else "fixed chain not found" if src is None
           else "differs from the fixed chain" if not filecmp.cmp(txt, src, shallow=False) else f"checkpoint left: {ckpt}")
    report(ok, f"restored {chain} = {src.name if src else '?'}", why)

# ---------------------------------------------------------------- summary
print()
if fails:
    print(f"{len(fails)} problem(s). Conclusion: fix these before submitting:")
    for f in fails:
        print("   -", f)
else:
    print("Conclusion: all checks passed, safe to run `bash submit_lsst_rerun.sh go`.")
