# Dataset guide

## A. Included demonstration data

`datasets/synthetic` contains 180 original generated images with known artificial labels, a CSV manifest and a data card. Each artificial subject occurs in only one split. This is immediately usable to demonstrate file ingestion, preprocessing, training, validation, test evaluation and inference.

Do not rename artificial pattern labels to normal, benign or malignant. There are no real patients, clinical diagnoses or cancer stages in these files.

## B. Real CT dataset: IQ-OTH/NCCD

Official download: https://data.mendeley.com/datasets/bhmdr45bh2/4

Click **Download All** on the dataset page and extract locally under `datasets/real/`. A download may require the dataset provider's own browser flow; this package does not bypass it.

The publisher describes 1,190 CT slices from 110 cases, grouped into normal, benign and malignant. The licence shown is **CC BY 4.0**. Cite the authors and preserve the source's licence terms. Inspect the actual archive and labels; do not assume the stated number matches the number of usable decoded images after validation.

Dataset citation: alyasriy, hamdalla; AL-Huseiny, Muayed (2023), “The IQ-OTH/NCCD lung cancer dataset”, Mendeley Data, V4, doi: 10.17632/bhmdr45bh2.4.

This dataset's three class labels are not Stage I, II, III and IV. Do not derive stages from filenames, class scores, lesion brightness or arbitrary thresholds.

### Prepare the manifest

Create `datasets/real/raw_manifest.csv` with these columns:

```csv
path,label,patient_id,source
images/example-normal.png,normal,VERIFIED_SUBJECT_001,IQ-OTH-NCCD-v4
images/example-benign.png,benign,VERIFIED_SUBJECT_002,IQ-OTH-NCCD-v4
images/example-malignant.png,malignant,VERIFIED_SUBJECT_003,IQ-OTH-NCCD-v4
```

These are schema examples, not real dataset files or identities. Replace them with actual relative paths, source labels and verified subject IDs. Different slices of the same patient must use exactly the same patient ID. Obtain a reliable mapping from dataset metadata or its maintainers; do not assume a filename or arbitrary image index is a unique patient. If reliable IDs cannot be established, do not claim a leakage-free patient-level evaluation.

The starter split utility requires at least five verified patients in every class. It creates a seeded approximate 60/20/20 split stratified by patient class. Keep augmentation and any data-driven preprocessing inside the training partition. Never put related slices or augmented copies into different splits.

```powershell
.\.venv\Scripts\python.exe scripts/split_manifest.py datasets/real/raw_manifest.csv
```

The generated `manifest.csv` stays beside the input so relative image paths work. The trainer rejects overlapping patient IDs, duplicate paths, and identical preprocessed images across splits. These checks catch common leakage but cannot prove that incorrectly supplied IDs represent different people.

### What can be measured?

The included evaluation reports image-level class metrics. Correlated CT slices mean this is not a patient-level clinical performance estimate. For a stronger study, aggregate by patient, evaluate confidence intervals, assess sensitivity/specificity and calibration, test external data and document exclusions. Do not repeatedly tune on the held-out test partition.

### Image format

The starter handles 8-bit PNG/JPEG/WEBP, EXIF orientation, grayscale conversion and fixed 32×32 resizing. It does not parse DICOM series or Hounsfield units. Use a dedicated, validated preprocessing pipeline for volumetric DICOM research rather than renaming DICOM files to PNG.

## C. More advanced CT research

The Cancer Imaging Archive's **LIDC-IDRI** collection is another research resource with CT scans and nodule annotations:
https://www.cancerimagingarchive.net/collection/lidc-idri/

It requires a different data preparation workflow and is not plug-and-play with this 2D image baseline. Radiologist nodule ratings must not automatically be treated as pathologically proven cancer or as TNM stages. Review the collection documentation, data terms and annotation definitions before use.

## Sources checked for this package

Dataset publisher, version 4: https://data.mendeley.com/datasets/bhmdr45bh2/4
TCIA collection listing: https://www.cancerimagingarchive.net/collection/lidc-idri/
NCI staging reference: https://www.cancer.gov/types/lung/patient/non-small-cell-lung-treatment-pdq
