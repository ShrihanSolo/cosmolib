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
n_ok = n_bad = 0
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
    n_ok += ok
    n_bad += not ok
    print(f"{'ok' if ok else 'WRONG':8s} {f'{bad}/{2 * n}':>18s}   {fp}")

print()
print(f"{n_ok} data vectors match their covariance order, {n_bad} do not.")
print("Conclusion:", "all good, safe to run chains." if n_bad == 0 else
      "rerun the cosmomake notebooks for the WRONG files before running chains.")
