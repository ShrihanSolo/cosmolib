#!/bin/bash
# Re-executes every LSST cosmomake notebook in place (2026-11-15 row-order fix), then checks the
# data-vector order against the covariances. Run from cosmolib/shrihan on a compute node, in the
# environment you normally run the cosmomake notebooks with (needs jupyter nbconvert):
#     cd ~/weaklens/cosmolib/shrihan && bash run_lsst_cosmomake.sh 2>&1 | tee run_lsst_cosmomake.log
# A notebook whose order assertion fails stops with an error and is listed at the end.
# The *_fixed notebooks are skipped: their outputs are identical to the patched fidA/dcrA ones.

failed=()
for nb in cosmomake_clean_lssty1_*.ipynb cosmomake_clean_lssty5_*.ipynb cosmomake_clean_lssty10_*.ipynb; do
    case "$nb" in *_fixed.ipynb) continue ;; esac
    echo "=== $nb"
    if ! jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=1800 "$nb" > /dev/null 2> "${nb%.ipynb}.nbconvert.out"; then
        echo "    FAILED (see ${nb%.ipynb}.nbconvert.out)"
        failed+=("$nb")
    elif ! grep -q "Covariance and datavector ordering match" "$nb"; then
        echo "    ran, but the order check did not print; inspect the notebook"
        failed+=("$nb")
    else
        echo "    ok"
    fi
done

echo
python check_dv_order.py
echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "All LSST cosmomake notebooks ran and passed the order check."
else
    echo "Notebooks needing attention: ${failed[*]}"
fi
