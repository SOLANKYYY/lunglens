# LungLens — Lung Screening Research Project

A VS Code-ready academic project inspired by the lung screening concept described in your brief. Independent project: not affiliated with 12 Sigma and not a reproduction of its proprietary algorithm.

**Start with the local demonstration below.** The package contains the complete website, Python API, training code for three framework options, a trained artificial-pattern baseline and 180 synthetic PNG images.

## What is implemented?

| Component | Status |
|---|---|
| Responsive HTML/CSS/JavaScript dashboard | Included |
| Scan preview, zoom, brightness, scripted samples | Included |
| Local Python image inference API | Included; tested with NumPy model |
| Training/validation/test pipeline and confusion matrix | Included |
| NumPy artificial-pattern model | Trained; weights and evaluation included |
| TensorFlow and PyTorch implementations | Included; optional dependencies; not runtime-tested in the build environment |
| Normal / benign / malignant research training | Supported after preparing a real dataset manifest; no trained cancer classifier included |
| Cancer stage learning cards (0–IV) | Included; educational NSCLC overview |
| Automatic cancer staging or clinical diagnosis | Not implemented; stage always “Not assessed” |
| Vercel static frontend configuration | Included; not yet deployed to Vercel |

## 1. Open in VS Code (Windows)

1. Extract `lunglens-vscode.zip` fully.
2. VS Code → **File → Open Folder** → select `lunglens-vscode` (the folder containing this README).
3. Install Python 3.11 and the Microsoft Python extension if needed.
4. Open **Terminal → New Terminal**, then run from this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m backend.server
```

Open **http://127.0.0.1:8000** in your browser. Keep the terminal running. Stop with Ctrl+C.

If `py` is unavailable but Python 3.11 is installed, use `python -m venv .venv` for the first command. The explicit virtual-environment Python commands avoid PowerShell activation-policy problems.

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m backend.server
```

Choose `.venv` with **Python: Select Interpreter** in VS Code. An F5 launch configuration is included.

## 2. Show a real code-to-output demonstration

1. Open **Datasets & model**. It should say **Local Python API connected**.
2. Download the **Round-pattern sample** or another bundled synthetic example.
3. In **Overview → Browse files**, select that downloaded PNG.
4. Click **Run local model**. The Python code computes class scores from the image pixels using the supplied weights.
5. Download the JSON model report. It records the framework, artificial-pattern class, scores, limitations and `stage: null`.
6. Open **Cancer stages** to explain Stage 0–IV as a separate educational reference.
7. Show `backend/train.py`, `backend/common.py`, `backend/inference.py` and `models/demo/evaluation.json` to faculty.

The original “Run demo analysis” on drawn sample cases is a **scripted UI simulation**. “Run local model” on an uploaded PNG is **actual computation using the trained synthetic-pattern baseline**. Neither is a cancer diagnosis.

## 3. Train and test the included dataset

The 180 images are already included under `datasets/synthetic/`:

| Split | Images | Per class |
|---|---:|---:|
| Training | 108 | 36 |
| Validation | 36 | 12 |
| Test | 36 | 12 |

Classes: `pattern_none`, `pattern_round`, `pattern_irregular`. They are not medical diagnoses.

Run the tested NumPy baseline:

```powershell
.\.venv\Scripts\python.exe -m backend.train --framework numpy --epochs 200 --output models/demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The training script uses training data for updates, validation loss to choose weights, and the test split for the final evaluation. It writes `metadata.json`, weights and `evaluation.json` with a confusion matrix, precision, recall and macro F1. Restart the Python server after retraining to load the new weights.

To regenerate the same synthetic dataset:

```powershell
.\.venv\Scripts\python.exe scripts/generate_demo_data.py
```

The included run scored **34/36 on the synthetic test set (94.44%)**. This measures recognition of generated shapes only. **Do not call it lung cancer detection accuracy.** No real CT images were used to train this included model.

### PyTorch option

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pytorch.txt
.\.venv\Scripts\python.exe -m backend.train --framework pytorch --epochs 100 --output models/pytorch-demo
.\.venv\Scripts\python.exe -m backend.server --model models/pytorch-demo
```

### TensorFlow option

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tensorflow.txt
.\.venv\Scripts\python.exe -m backend.train --framework tensorflow --epochs 100 --output models/tensorflow-demo
.\.venv\Scripts\python.exe -m backend.server --model models/tensorflow-demo
```

Use **one framework at a time**. Both optional implementations use the same simple linear softmax architecture for a transparent comparison. This is not a CNN, segmentation model or nodule detector. The framework versions are dependency ranges, not a frozen reproducibility lockfile. After a successful installation, record your environment with `python -m pip freeze > requirements-lock.txt` and cite it in your submission.

## 4. Real CT training data

See [docs/DATASETS.md](docs/DATASETS.md) for the official **IQ-OTH/NCCD** download, licence, citations, a manifest example and patient-level split instructions. Real CT files are not bundled. They are separate from the synthetic demonstration.

For a prepared real manifest with verified patient IDs:

```powershell
.\.venv\Scripts\python.exe scripts/split_manifest.py datasets/real/raw_manifest.csv
.\.venv\Scripts\python.exe -m backend.train --framework pytorch --manifest datasets/real/manifest.csv --epochs 100 --output models/research-pytorch
.\.venv\Scripts\python.exe -m backend.server --model models/research-pytorch
```

Install `requirements-pytorch.txt` first. The real manifest uses `normal`, `benign`, `malignant` labels. Use source annotations; do not invent diagnoses, patient IDs or stages. A research prediction is not a clinical finding and arbitrary non-CT inputs may also receive a class.

## 5. Push to GitHub

Create a new empty repository on GitHub. Do not initialize it with another README. Then run in the VS Code terminal:

```bash
git init
git add .
git commit -m "Add LungLens academic project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/lunglens.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your account name. The package includes no existing `.git` history, hosted Site IDs, tokens or credentials. `.gitignore` excludes virtual environments, local secrets, real datasets and optional research model directories. It intentionally keeps the small synthetic dataset and NumPy demo model for reproducibility.

## 6. Later deploy the frontend to Vercel

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Import your GitHub repository, use **Other** as the framework and **web** as the output directory. No frontend build is needed.

**Vercel static hosting publishes the interface and scripted demo, not the running local Python model.** The package's Python server is for local development. A public inference service requires a separate production backend and integration; do not claim model inference is live merely because the website is deployed.

## Project structure

```text
lunglens-vscode/
  web/                  Website; Vercel publishes this directory
    index.html          Dashboard, reports, stages, datasets
    styles.css          Responsive design
    app.js              Viewer, scripted samples, session reports
    project.js          Stage cards and local Python API integration
    demo_samples/       Three downloadable synthetic test images
  backend/
    common.py           Shared preprocessing, split checks, evaluation
    train.py            NumPy / PyTorch / TensorFlow training
    inference.py        Trusted model loading and prediction
    server.py           Local website + image API
  datasets/synthetic/   180 PNGs, manifest.csv, dataset card
  models/demo/          Tested NumPy weights, metadata, evaluation
  scripts/              Synthetic generation and real-data splitting
  tests/                Inference, leakage and HTTP checks
  docs/                 Datasets, deployment and faculty presentation guide
  .vscode/              Recommended extensions and debug launch settings
  vercel.json           Static deployment configuration
```

## Limitations

This is an educational software project. It does not establish diagnostic accuracy, localization, calibrated cancer risk or cancer stage. Synthetic examples are deliberately simple. Real-data research requires stronger models, verified labels, appropriate evaluation and specialist review before any clinical use. Stage education is based on NCI's NSCLC reference; it is not a complete staging rule engine.

## Sources

- IQ-OTH/NCCD dataset: https://data.mendeley.com/datasets/bhmdr45bh2/4
- NCI NSCLC staging: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
- TensorFlow image classification: https://www.tensorflow.org/tutorials/images/classification
- PyTorch quickstart: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- Vercel configuration: https://vercel.com/docs/project-configuration/vercel-json

Verification details: [docs/VERIFICATION.md](docs/VERIFICATION.md).
