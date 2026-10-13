# Publish from the existing Quantum-Circuit-Optimization-Advisor repository

This complete ZIP contains one Quantum-Circuit-Optimization-Advisor project folder. Use its contents inside your existing clone of that same repository. Keep the existing Git history. If you have local edits not present in GitHub, retain a copy before replacing files. Website files belong in docs; original project artifacts stay at the root.

## Upload to your current GitHub repository

Repository: https://github.com/Rishithachappidi/Quantum-Circuit-Optimization-Advisor

If the local folder is already a Git clone:

git add docs tools
git commit -m "Add browser dashboard for GitHub Pages"
git push

If the Downloads folder is not a Git clone, use GitHub Desktop to clone that existing repository, then copy docs and tools into the cloned folder and commit/push. This uses the same repository, not a new one. Or upload docs and tools through GitHub's Add file > Upload files. The data folder contains more than 100 files, so browser uploads may need batches while preserving docs/data/.

## Enable GitHub Pages

Settings > Pages > Source: Deploy from a branch > Branch: main > Folder: /docs > Save.

Expected public address after successful deployment:
https://Rishithachappidi.github.io/Quantum-Circuit-Optimization-Advisor/

## What to edit

- docs/index.html: page content and controls.
- docs/style.css: appearance.
- docs/app.js: interface behavior and output presentation.
- docs/engine.js: browser validation, exposure equations and recommendation rules.
- docs/data/: generated circuit and prediction records; avoid hand-editing result numbers.
- app.py and the original notebook: original Python/Streamlit project.
- tools/build_pages_data.py: generate real Qiskit choices and original-model predictions into docs/data/.

## Rebuild dashboard data after changing circuits or the trained model

Install tools/requirements-pages.txt in a separate Python environment. The original Streamlit requirements.txt is not changed.

python -m pip install -r tools/requirements-pages.txt
python tools/build_pages_data.py

The generator reuses existing JSON choices. To fully regenerate after changing source/model, remove the generated docs/data JSON files first. The original notebook is the training workflow; modifying the workbook alone does not retrain the model. After regeneration, commit docs/data along with the corresponding code/model changes.

Optional validation (Node.js): node tools/check_pages.mjs

Changing Python/model source does not automatically regenerate the browser data. Commit regenerated JSON to make those changes available on GitHub Pages. HTML/CSS/JS edits publish directly on the configured branch.

## Dashboard scope

All nine families allow 2–10 total qubits. Bell uses one original Bell pair plus idle measured qubits in larger registers. Simon uses two equal registers plus one idle qubit for odd totals; 2/3-total-qubit cases are degenerate. Expanded selections do not represent a newly trained model. Original model metrics and dataset are retained and out-of-range predictions are marked as indicative. Custom circuits receive structural analysis only. No real hardware execution or new noisy simulation runs occur on the public page.
