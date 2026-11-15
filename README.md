# LungLens — Lung Cancer Detection Research

Upload a de-identified CT image and run a **saved model trained on real IQ-OTH/NCCD CT images**. The default website now uses real inference; it does not use the old synthetic-pattern demonstration.

## Live app

https://lunglens-seven.vercel.app

Choose one of six original test images, or download it and upload it again. Confirm it is a de-identified CT image, then click **Run trained model**. Download the resulting JSON report. The app displays normal / benign / malignant **dataset class predictions**, not a diagnosis. Scores are raw kernel regression outputs, not cancer probabilities. Cancer stage and severity are not assessed.

## Dataset and attribution

- Kaggle: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
- Publisher and licence: https://data.mendeley.com/datasets/bhmdr45bh2/4
- Downloaded author mirror: https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset
- Source commit: `edcdd5a73e8e92377d2a47e69c0b9a551e1c5ce3`
- alyasriy, hamdalla; AL-Huseiny, Muayed (2023), *The IQ-OTH/NCCD lung cancer dataset*, Mendeley Data V4, DOI **10.17632/bhmdr45bh2.4**, **CC BY 4.0**.
- H. F. Al-Yasriy et al., *Diagnosis of Lung Cancer Based on CT Scans Using CNN*, IOP Conference Series: Materials Science and Engineering 928 (2020).
- H. F. Kareem et al., *Evaluation of SVM performance in the detection of lung cancer in marked CT scan dataset*, IJEECS 21(3), 1731–1738 (2021), DOI 10.11591/ijeecs.v21.i3.pp1731-1738.

Kaggle download was unavailable during this build, so the publisher-linked author mirror was used. The downloaded release contains **1,097 JPEGs** (416 normal, 120 benign, 561 malignant), despite the publisher's description mentioning 1,190 images. No artificial or augmented images were used to train this model. The six bundled examples are byte-identical original JPEGs; their source filenames and hashes are in `web/real_samples/samples.json`. They are shared under the dataset's CC BY 4.0 licence, separate from the software licence.

## Model and measured results

RBF kernel ridge classifier, 64×64 grayscale central crops. NumPy runtime; NumPy, SciPy and scikit-learn training. This is a classical machine-learning baseline, not a CNN or lesion-localization model.

| Partition | Images |
|---|---:|
| Training | 657 |
| Validation | 220 |
| Held-out test | 220 |

Validation macro F1 selects gamma and regularization. Only training images fit the weights. Test evaluation follows parameter selection: **accuracy 85.91%, macro F1 0.8125**. Malignant-class recall is 88.39%; 13 malignant test slices were predicted normal and 3 benign. See `models/ct/evaluation.json` for the complete confusion matrix and class metrics.

**Critical evaluation limit:** filenames are sequential image numbers, not patient IDs. Exact decoded duplicates and perceptual hashes within Hamming distance 4 are grouped before splitting (254 similarity groups). Group overlap is checked, but unrecognized related slices and patient overlap may remain. These are exploratory image-level metrics, not patient-independent validation, calibrated risk, or evidence of clinical performance. Arbitrary non-CT pictures can receive a prediction. Low-contrast and malformed images are rejected; this is not a validated CT detector.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m backend.ct_server
```

Open http://127.0.0.1:8000. The model is loaded from `models/ct/weights.npz` using `allow_pickle=False`; no retraining is needed to restore it.

## Reproduce training

```powershell
git clone https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset.git ../iqoth-source
git -C ../iqoth-source checkout edcdd5a73e8e92377d2a47e69c0b9a551e1c5ce3
python -m pip install -r requirements-train.txt
python -m scripts.train_ct --source ../iqoth-source
python -m unittest discover -s tests -v
```

Seed 42. `models/ct/manifest.json` records every source file, split, similarity group and SHA256. `metadata.json` records the source commit, selected parameters and weight hash. Original training images remain outside the app checkout. Training overwrites the saved CT model and regenerates the six samples using filename order, without selecting for successful predictions.

## Hosting

The existing Vercel project serves `web/` plus Python functions `api/health.py` and `api/predict.py`. The Python functions restore the bundled model and execute inference. This is a live model API, not a static mock. Runtime dependencies are NumPy and Pillow; no API key, external AI provider or separate paid model host is needed. Uploads are limited to 4 MB and processed in memory; the app does not persist them or log their contents.

The legacy synthetic training scripts and old tests remain for reproducibility but are not loaded by the new default interface. Use `backend.ct_server` for the new app, not the legacy `backend.server`.

## Medical scope

This is an educational research project, not a medical device. A result cannot diagnose or exclude cancer, recommend treatment, or determine seriousness. Stage requires clinical assessment and information about tumor extent, nodes and spread; the image dataset has no per-image stage labels. Reference: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
