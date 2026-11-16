"""
make_lsst_2pt.py

Writes every LSST 2pt data-vector fits file used by the cosmosis inis, replacing the per-variant
cosmomake_clean_lssty*.ipynb notebooks. The recipe is the one in those notebooks (2026-11-15):

  * xi+/xi-/npairs from clean_sims/<YEAR>/<kind>2pt/A/results (treecorr outputs), 5 source bins,
    20 angular bins; angle columns and fits layout from old_runs/giulia_2pt/giulia_2pt_example.fits
    (4-bin template, extended to 5 bins);
  * scaled imprints: xi = xi_fid + A * (xi_imprint - xi_fid), with A a number (x3, x9) or, for the
    PSF "rn" variants, A = factor * ratio(theta), ratio = xi+_true / xi+_pred of the DELVE PSF
    regressor (map_correlations/regressor/twopoint), interpolated in ln(theta);
  * rows sorted into CosmoCov order (11,12,...,15,22,...,55; angle fastest) and checked against the
    covariance row by row (the bug fixed on 2026-11-15: the 5th-bin pairs used to sit at the end);
  * covariance = Gaussian + non-Gaussian CosmoCov blocks, n(z) from <YEAR>_Data, sigma_e = 0.26,
    n_gal per bin 2.0 / 4.2 / 5.4 arcmin^-2 for Y1 / Y5 / Y10.

Before overwriting, each output is compared with the file already on disk (same bin pair and angle),
so the log says whether the values changed or only the row order did.

Run from cosmolib/shrihan:
    python make_lsst_2pt.py              # build all variants
    python make_lsst_2pt.py --list       # list variants and output files only
    python make_lsst_2pt.py --only LSSTY10_dcrimprint2pt9x_A_2pt LSSTY1_fiducial2pt_A_2pt
    python make_lsst_2pt.py --plots      # also make the twopoint diagnostic figures (slow)
Needs numpy and astropy (and the local twopoint package for --plots).
"""
import argparse
import glob
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent            # cosmolib/shrihan
WEAKLENS = HERE.parent.parent                     # ~/weaklens
SIMS = WEAKLENS / "clean_sims"
COSMOCOW_OUT = WEAKLENS / "cosmocow" / "covs" / "output"
TEMPLATE = HERE / "old_runs" / "giulia_2pt" / "giulia_2pt_example.fits"
REGRESSOR = WEAKLENS / "map_correlations" / "regressor" / "twopoint"

NBIN, NANG = 5, 20
NPAIR = NBIN * (NBIN + 1) // 2                     # 15
NROW = NPAIR * NANG                                # 300 rows each for xi+ and xi-
SIGMA_E = 0.26
NGAL = {"LSSTY1": 2.0, "LSSTY5": 4.2, "LSSTY10": 5.4}
COVNAME = {"LSSTY1": "lssty1", "LSSTY5": "lssty5", "LSSTY10": "lssty10"}

# (year, kind, scaling, output name)
#   kind:    'fiducial' or an imprint ('psf', 'resp', 'dcr', 'cs')
#   scaling: None (imprint as measured), a number A, or ('rn', A) for A * regressor ratio(theta)
VARIANTS = []
for _y in ("LSSTY1", "LSSTY5", "LSSTY10"):
    VARIANTS.append((_y, "fiducial", None, f"{_y}_fiducial2pt_A_2pt"))
    for _k in ("psf", "resp", "dcr", "cs"):
        VARIANTS.append((_y, _k, None, f"{_y}_{_k}imprint2pt_A_2pt"))
VARIANTS += [
    ("LSSTY1", "dcr", 3.0, "LSSTY1_dcrimprint2pt3x_A_2pt"),
    ("LSSTY1", "dcr", 9.0, "LSSTY1_dcrimprint2pt9x_A_2pt"),
    ("LSSTY10", "resp", 3.0, "LSSTY10_respimprint2pt3x_A_2pt"),
    ("LSSTY10", "cs", 3.0, "LSSTY10_csimprint2pt3x_A_2pt"),
    ("LSSTY10", "dcr", 3.0, "LSSTY10_dcrimprint2pt3x_A_2pt"),
    ("LSSTY10", "dcr", 9.0, "LSSTY10_dcrimprint2pt9x_A_2pt"),
    ("LSSTY10", "psf", ("rn", 1.0), "LSSTY10_psfimprint2ptrn_A_2pt"),
    ("LSSTY10", "psf", ("rn", 3.0), "LSSTY10_psfimprint2ptrn3x_A_2pt"),
    ("LSSTY10", "psf", ("rn", 9.0), "LSSTY10_psfimprint2ptrn9x_A_2pt"),
]


def pairs():
    # Bin pairs (b1 <= b2) in CosmoCov order: 11, 12, ..., 15, 22, ..., 55.
    return [(i, j) for i in range(1, NBIN + 1) for j in range(i, NBIN + 1)]


def load_measurement(year, kind):
    # treecorr outputs written by cross_sims.py: files indexed r{j}{i} with j <= i (0-based bins).
    run_dir = SIMS / year / ("fiducial2pt" if kind == "fiducial" else f"{kind}imprint2pt") / "A" / "results"
    xip, xim, theta, npairs = {}, {}, {}, {}
    for i in range(NBIN):
        for j in range(i + 1):
            key = (j + 1, i + 1)
            xip[key] = np.load(run_dir / f"xip_corr_r{j}{i}.npy")
            xim[key] = np.load(run_dir / f"xim_corr_r{j}{i}.npy")
            theta[key] = np.load(run_dir / f"theta_corr_r{j}{i}.npy")
            with open(run_dir / f"gg_r{j}{i}.pickle", "rb") as f:
                npairs[key] = pickle.load(f).npairs
    return dict(xip=xip, xim=xim, theta=theta, npairs=npairs, run_dir=run_dir)


def regressor_ratio(theta):
    # xi+_true / xi+_pred of the DELVE PSF regressor, on the measurement's theta grid (as in the rn notebooks).
    load = lambda q, k: np.load(REGRESSOR / f"DELVE_de1_{q}_{k}.npy")
    th_t, xi_t = load("r", "true"), load("xip", "true")
    th_p, xi_p = load("r", "pred"), load("xip", "pred")
    return np.interp(np.log(theta), np.log(th_t), xi_t) / np.interp(np.log(theta), np.log(th_p), xi_p)


def apply_scaling(meas, fid, scaling):
    if scaling is None:
        return meas["xip"], meas["xim"]
    if isinstance(scaling, tuple):
        amp = scaling[1] * regressor_ratio(meas["theta"][(1, 1)])
    else:
        amp = float(scaling)
    xip = {k: fid["xip"][k] + amp * (meas["xip"][k] - fid["xip"][k]) for k in pairs()}
    xim = {k: fid["xim"][k] + amp * (meas["xim"][k] - fid["xim"][k]) for k in pairs()}
    return xip, xim


def covariance_file(year):
    # Same as the notebooks' `cat .../out_cov_<year>_shrihan/l* > covdata_<year>.txt`, if the CosmoCov output is here.
    name = COVNAME[year]
    out = HERE / f"covdata_{name}.txt"
    parts = sorted(glob.glob(str(COSMOCOW_OUT / f"out_cov_{name}_shrihan" / "l*")))
    if parts:
        with open(out, "wb") as w:
            for p in parts:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
        print(f"  covariance: concatenated {len(parts)} CosmoCov files -> {out.name}")
    else:
        print(f"  covariance: CosmoCov output not found, using existing {out.name}")
    return out


def read_covariance(covfile):
    # Exactly plot.get_cov (row by row, so entries listed as both (i,j) and (j,i) resolve the same way):
    # column 0/1 = indices, 8 = Gaussian, 9 = non-Gaussian. Also returns the (bin1, bin2) of each index.
    data = np.genfromtxt(covfile)
    n = int(np.max(data[:, 0])) + 1
    cov_g, cov_ng = np.zeros((n, n)), np.zeros((n, n))
    order = {}
    for r in data:
        i, j = int(r[0]), int(r[1])
        cov_g[i, j] = cov_g[j, i] = r[8]
        cov_ng[i, j] = cov_ng[j, i] = r[9]
        order.setdefault(i, (int(r[4]) + 1, int(r[5]) + 1))
    return cov_g + cov_ng, order


def build_tables(template, xip, xim, npairs):
    # Rows in CosmoCov order. Angle columns (ANG differs between template bin pairs by <0.01%) as in the notebooks:
    # pairs in the 4-bin template keep their own template row; pair (b1, 5) takes the row of the b1-th template
    # pair (11, 12, 13, 14, 22) at the same angular bin, both for xi+ and xi- (the notebooks used the xi+ rows).
    tp = template["xip"].data
    tpairs = [(int(tp["BIN1"][k]), int(tp["BIN2"][k])) for k in range(0, len(tp), NANG)]
    tabs = {}
    for ext, xi in (("xip", xip), ("xim", xim)):
        t = template[ext].data
        rec = np.recarray(NROW, dtype=t.dtype)
        r = 0
        for (b1, b2) in pairs():
            for a in range(NANG):
                if b2 < NBIN:
                    src, row = t, tpairs.index((b1, b2)) * NANG + a
                else:
                    src, row = tp, (b1 - 1) * NANG + a
                assert int(src["ANGBIN"][row]) == a
                for c in ("ANGLEMIN", "ANGLEMAX", "ANG"):
                    rec[c][r] = src[c][row]
                rec["BIN1"][r], rec["BIN2"][r], rec["ANGBIN"][r] = b1, b2, a
                rec["VALUE"][r] = xi[(b1, b2)][a]
                rec["NPAIRS"][r] = npairs[(b1, b2)][a]
                r += 1
        tabs[ext] = rec
    return tabs


def rebuild_hdu(original_hdu, new_data):
    # New table HDU with the template's column formats/units and non-column header cards (as in the notebooks).
    col = {c.name: c for c in original_hdu.columns}
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format=col[n].format, unit=col[n].unit, array=new_data[n]) for n in new_data.dtype.names])
    for key, val in original_hdu.header.items():
        if key not in hdu.header and key not in ("NAXIS1", "NAXIS2", "TFIELDS"):
            try:
                hdu.header[key] = val
            except Exception:
                pass
    hdu.name = original_hdu.name
    hdu.header["EXTNAME"] = original_hdu.name.lower()
    hdu.header["N_ZBIN_1"] = NBIN
    hdu.header["N_ZBIN_2"] = NBIN
    return hdu


def check_order(order, tabs):
    # Every data row must have the (bin1, bin2) of the covariance row with the same index, angle fastest.
    assert len(order) == 2 * NROW, (len(order), 2 * NROW)
    for off, ext in ((0, "xip"), (NROW, "xim")):
        d = tabs[ext]
        for i in range(NROW):
            assert order[off + i] == (d["BIN1"][i], d["BIN2"][i]), (ext, i, order[off + i], d["BIN1"][i], d["BIN2"][i])
            assert d["ANGBIN"][i] == i % NANG, (ext, i, d["ANGBIN"][i])


def compare_with_existing(path, tabs):
    # Compare with the file being replaced, matching rows by (bin1, bin2, angbin), so a pure reordering shows as 0.
    if not path.exists():
        return "new file"
    with fits.open(path) as old:
        rel = 0.0
        for ext in ("xip", "xim"):
            o = {(int(r["BIN1"]), int(r["BIN2"]), int(r["ANGBIN"])): float(r["VALUE"]) for r in old[ext].data}
            new = np.array([v for v in tabs[ext]["VALUE"]])
            ref = np.array([o[(int(b1), int(b2), int(a))] for b1, b2, a in
                            zip(tabs[ext]["BIN1"], tabs[ext]["BIN2"], tabs[ext]["ANGBIN"])])
            rel = max(rel, np.max(np.abs(new - ref)) / np.max(np.abs(ref)))
    return "values identical to the old file (only the row order changed)" if rel < 1e-12 else \
        f"VALUES CHANGED vs the old file: max |new-old|/max|old| = {rel:.2e}"


def make(year, kind, scaling, name, cache, plots=False):
    print(f"\n=== {name}")
    if year not in cache:
        cache[year] = {"fid": load_measurement(year, "fiducial")}
        cache[year]["cov"], cache[year]["order"] = read_covariance(covariance_file(year))
    fid = cache[year]["fid"]
    meas = fid if kind == "fiducial" else load_measurement(year, kind)
    print(f"  input: {meas['run_dir']}" + (f", scaling {scaling}" if scaling is not None else ""))
    xip, xim = apply_scaling(meas, fid, scaling)

    ffile = fits.open(TEMPLATE)
    del ffile[4:6]          # gamma_t, wtheta
    del ffile[-1]           # nz_lens  -> [PRIMARY, COVMAT, xip, xim, nz_source]
    tabs = build_tables(ffile, xip, xim, meas["npairs"])
    check_order(cache[year]["order"], tabs)
    print("  row order matches the covariance")
    ffile[2] = rebuild_hdu(ffile[2], tabs["xip"])
    ffile[3] = rebuild_hdu(ffile[3], tabs["xim"])

    for k in ("STRT_2", "NAME_2", "STRT_3", "NAME_3"):
        ffile[1].header.pop(k, None)
    ffile[1].header["STRT_1"] = NROW
    ffile[1].header["NAXIS1"] = 2 * NROW
    ffile[1].header["NAXIS2"] = 2 * NROW
    ffile[1].data = cache[year]["cov"]

    data_dir = SIMS / year / f"{year}_Data"
    ztomo = np.load(data_dir / f"ztomo_{year}.npy")
    nztomo = np.load(data_dir / f"nztomo_{year}.npy")
    zhead = ffile[4].header
    zhead["NBIN"] = NBIN
    zhead["NZ"] = 400
    for i in range(NBIN):
        zhead[f"NGAL_{i + 1}"] = NGAL[year]
        zhead[f"SIG_E_{i + 1}"] = SIGMA_E
    znew = np.recarray(len(ztomo), dtype=[("Z_LOW", float), ("Z_MID", float), ("Z_HIGH", float)] +
                       [(f"BIN{i + 1}", float) for i in range(NBIN)])
    znew["Z_MID"] = ztomo
    znew["Z_LOW"] = ztomo - ztomo[0]      # as in the notebooks; cosmosis reads Z_MID
    znew["Z_HIGH"] = ztomo + ztomo[0]
    for i in range(NBIN):
        znew[f"BIN{i + 1}"] = nztomo[i]
    ffile[4].data = znew

    out_dir = HERE / name
    out = out_dir / f"{name}.fits"
    print(f"  {compare_with_existing(out, tabs)}")
    os.makedirs(out_dir / "figures", exist_ok=True)
    ffile.writeto(out, overwrite=True)
    print(f"  wrote {out.relative_to(HERE)}")

    if plots:
        import matplotlib as mpl
        mpl.use("Agg")
        from twopoint import twopoint
        T = twopoint.TwoPointFile.from_fits(str(out))
        T.plots(str(out_dir / "figures" / name), colormap="viridis", savepdf=False, latex=False, plot_spectrum=True,
                plot_kernel=True, plot_cov=True, cov_vmin=None, sharey=True, save_pickle=False, load_pickle=False,
                remove_pickle=True, label_legend="", blind_yaxis=False, shade_until=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="output names to build (default: all)")
    ap.add_argument("--list", action="store_true", help="list the variants and exit")
    ap.add_argument("--plots", action="store_true", help="also make the twopoint figures")
    args = ap.parse_args()

    todo = [v for v in VARIANTS if not args.only or v[3] in args.only]
    if args.only:
        missing = set(args.only) - {v[3] for v in todo}
        if missing:
            raise SystemExit(f"unknown variant(s): {sorted(missing)}")
    if args.list:
        for y, k, s, n in todo:
            print(f"{n:38s} {y:8s} {k:9s} {'' if s is None else s}")
        return

    cache, failed = {}, []
    for v in todo:
        try:
            make(*v, cache, plots=args.plots)
        except Exception as e:          # keep going, report at the end
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(v[3])
    print(f"\n{len(todo) - len(failed)} of {len(todo)} data vectors written.")
    print("Conclusion:", "all built and in covariance order." if not failed else f"failed: {failed}")


if __name__ == "__main__":
    main()
