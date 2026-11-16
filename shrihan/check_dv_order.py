"""
check_dv_order.py

Checks that every 2pt data-vector fits file in this directory has its xi+/xi- rows in the
same (bin1, bin2, angbin) order as the CosmoCov covariance it is paired with.
cosmosis builds the theory vector from the BIN1/BIN2/ANGBIN columns, so a mismatch does not
scramble data vs theory, but it pairs each data row with the wrong covariance row.

Run from cosmolib/shrihan:   python check_dv_order.py
Needs numpy and astropy.
"""
import glob
import numpy as np
from astropy.io import fits

# Which CosmoCov text file each survey's data vectors use (same files the cosmomake notebooks read).
COVFILES = {
    "DES": "covdata_desy3.txt",
    "DELVE": "covdata_delve.txt",
    "LSSTY1": "covdata_lssty1.txt",
    "LSSTY5": "covdata_lssty5.txt",
    "LSSTY10": "covdata_lssty10.txt",
}
NANG = 20  # angular bins per bin pair


def cov_order(covfile):
    # CosmoCov rows: column 0 = data-vector index, columns 4/5 = source bins (0-based).
    c = np.loadtxt(covfile, usecols=(0, 4, 5))
    order = {}
    for i, b1, b2 in c:
        order.setdefault(int(i), (int(b1) + 1, int(b2) + 1))
    return order


orders = {s: cov_order(f) for s, f in COVFILES.items()}

# Data vectors read by some ini (2PT_FILE = shrihan/<dir>/<file>.fits); others are leftovers no chain uses.
used = set()
for ini in glob.glob("ini_files/*.ini"):
    for line in open(ini):
        if line.startswith("2PT_FILE"):
            used.add(line.split("=", 1)[1].strip().replace("shrihan/", "", 1))
n_ok = n_bad = n_unused = 0
print(f"{'result':8s} {'rows out of place':>18s}   file")
for fp in sorted(glob.glob("*_2pt*/*.fits")):
    survey = fp.split("_")[0]
    if survey not in orders:
        continue
    order = orders[survey]
    with fits.open(fp) as h:
        xip, xim = h["xip"].data, h["xim"].data
    n = len(xip)
    if len(order) != 2 * n:
        print(f"{'SIZE?':8s} {'cov ' + str(len(order)) + ' vs dv ' + str(2 * n):>18s}   {fp}")
        n_bad += 1
        continue
    bad = 0
    for off, d in ((0, xip), (n, xim)):
        for i in range(n):
            if order[off + i] != (d["BIN1"][i], d["BIN2"][i]) or d["ANGBIN"][i] != i % NANG:
                bad += 1
    ok = bad == 0
    in_use = fp in used
    n_ok += ok
    n_bad += (not ok) and in_use
    n_unused += (not ok) and not in_use
    label = "ok" if ok else ("WRONG" if in_use else "unused")
    print(f"{label:8s} {f'{bad}/{2 * n}':>18s}   {fp}" + ("" if ok or in_use else "   (wrong order, but no ini reads it)"))

print()
print(f"{n_ok} data vectors match their covariance order; {n_bad} used by an ini do not; "
      f"{n_unused} leftover files no ini reads are out of order.")
print("Conclusion:", "all data vectors used by the inis are in order, safe to run chains." if n_bad == 0 else
      "rebuild the WRONG files (python make_lsst_2pt.py) before running chains.")
