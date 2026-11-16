"""
check_lsst_inis.py

For every LSST data vector built by make_lsst_2pt.py, finds the cosmosis ini that reads it and checks:
  1. exactly one ini in ini_files/ has that 2PT_FILE, and the file exists;
  2. the ini differs from its year's fidA ini only in 2PT_FILE and the output filename
     (so priors, values, scale cuts, sampler and CAMB settings are the same within a year);
  3. the output chain name is unique.
Then prints the ini -> chain table to run.

Run from cosmolib/shrihan:   python check_lsst_inis.py
"""
import re
from pathlib import Path

from make_lsst_2pt import VARIANTS

HERE = Path(__file__).resolve().parent
INI_DIR = HERE / "ini_files"
YEAR_TAG = {"LSSTY1": "y1", "LSSTY5": "y5", "LSSTY10": "y10"}
VARYING = ("2PT_FILE", "filename = output")


def read(p):
    return [l.rstrip() for l in open(p)]


def field(lines, prefix):
    hits = [l.split("=", 1)[1].strip() for l in lines if l.startswith(prefix)]
    return hits[0] if hits else None


inis = {p: read(p) for p in sorted(INI_DIR.glob("lsst-*-clean-*.ini"))}
by_2pt = {}
for p, lines in inis.items():
    by_2pt.setdefault(field(lines, "2PT_FILE"), []).append(p)

problems, rows, outputs = [], [], {}
for year, kind, scaling, name in VARIANTS:
    want = f"shrihan/{name}/{name}.fits"
    found = by_2pt.get(want, [])
    if len(found) != 1:
        problems.append(f"{name}: {len(found)} inis read it ({[p.name for p in found]})")
        continue
    ini = found[0]
    if not (HERE.parent / want).exists():
        problems.append(f"{name}: {want} does not exist yet (run make_lsst_2pt.py)")
    fid = INI_DIR / f"lsst-{YEAR_TAG[year]}-clean-fidA.ini"
    a = [l for l in inis[fid] if l.strip() and not l.startswith(VARYING)]
    b = [l for l in inis[ini] if l.strip() and not l.startswith(VARYING)]
    if a != b:
        extra = sorted(set(a) ^ set(b))
        problems.append(f"{ini.name}: differs from {fid.name} beyond 2PT_FILE/filename: {extra}")
    out = re.sub(r"^output/", "", field(inis[ini], "filename = output"))
    if out in outputs:
        problems.append(f"{ini.name} and {outputs[out]} both write {out}")
    outputs[out] = ini.name
    rows.append((name, ini.name, out))

print(f"{'data vector':38s} {'ini':32s} chain")
for r in rows:
    print(f"{r[0]:38s} {r[1]:32s} {r[2]}")
unused = [p.name for p in inis if p.name not in {r[1] for r in rows}]
print(f"\ninis not used by any current data vector: {unused}")
print(f"\n{len(rows)} of {len(VARIANTS)} data vectors have a matching ini.")
print("Conclusion:", "all inis consistent." if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
