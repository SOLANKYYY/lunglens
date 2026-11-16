# LungLens: faculty demonstration

Class: AIML B
Subject: Essentials of Deep Learning
Topic: Lung Cancer Detection

## Team

1. **Solanki Om Narendra — 24BTRCL190 — Team Leader**
2. Jayansh Jain — 24BTRCL088
3. Jainish V Jain — 24BTRCL087
4. Amin Heet Deepakkumar — 24BTLCL002

## A simple opening

“Our project, LungLens, classifies individual chest CT images into normal, benign and malignant dataset categories. We trained a convolutional neural network on real CT images and deployed it as a web application. Its percentages describe the model's preference for each class. It does not diagnose cancer or assign a cancer stage.”

## Suggested presentation roles

- Solanki Om: introduce the problem, coordinate the live demonstration and explain limitations.
- Jayansh: explain CT, the dataset and why train/test separation matters.
- Jainish: explain convolution, ReLU, pooling, training loss and the CNN architecture.
- Amin: explain the technology stack, evaluation and future staging work.

These are suggested speaking roles, not claims about who implemented each component.

## Explain the model in one minute

A CT image contains pixel intensities. The program converts it to grayscale, removes 8% at each border, resizes it to 64 by 64 pixels, and divides intensities by 255. Three convolution layers learn visual patterns. ReLU adds a nonlinear operation and max pooling reduces spatial size. A dense layer combines learned features and outputs three logits. Softmax converts the logits into three scores summing to 100%. The largest score selects the class. The CNN has 285,699 trainable parameters.

The original JPEGs provide real data. We did not generate synthetic training images. We trained with PyTorch, Adam, a batch size of 32 and class-weighted cross entropy. We ran 40 epochs, selecting epoch 12 because it had the lowest validation loss. We saved the trained weights and exported the network to ONNX. Vercel restores it with ONNX Runtime, so the site really computes predictions.

## What to demonstrate

1. Open https://lunglens-seven.vercel.app and show the connected CNN status.
2. Select the first malignant dataset example, or download and upload it. Confirm educational use of a de-identified CT image.
3. Run the model. This example gives approximately 97.7% malignant, 1.7% normal and 0.6% benign scores. Read the result explanation.
4. Say: “97.7% is the model's malignant class score. It is not 97.7% of the lung affected, a validated cancer probability or a stage.”
5. Download the JSON report and show its model version, scores, explanation and null stage.
6. Try a normal or benign example. A prediction can disagree with the supplied label. Do not hide errors.
7. Open the separate NSCLC-Radiomics stage explorer. Select a case and explain that its stage comes from a published clinical record, not this image model.

## Honest results

The fixed split contains 657 training, 220 validation and 220 test images. The CNN classified 168/220 test images correctly (76.4%) and had macro F1 0.640. It recognized 97/112 malignant images and only 7/24 benign images. The earlier classical kernel baseline scored 85.9% accuracy. Deep learning did not outperform the baseline on this experiment. The small dataset, imbalance and overfitting are important limitations.

The files do not contain reliable patient IDs. Exact and perceptually similar images are grouped before the split, but patient overlap may remain. The score is an exploratory image-level result, not clinical or patient-independent accuracy.

## Faculty questions

**What is CT?** Computed tomography reconstructs cross-sectional images using X-rays from different angles. We use one axial chest slice at a time.

**How is CT different from a chest X-ray?** A plain X-ray projects overlapping anatomy onto a picture. CT reconstructs slices through the anatomy.

**What is DICOM?** A medical imaging format containing image data and acquisition metadata. Our app takes exported JPEG/PNG/WEBP images and does not process a complete DICOM volume.

**What is the difference between benign and malignant?** Benign describes non-cancerous disease in the source labels. Malignant describes cancerous disease. A model category alone cannot confirm either diagnosis.

**Why use validation data?** Training fits weights. Validation selects the checkpoint without choosing it on test accuracy. Testing estimates performance after selection.

**Why are scores not calibrated cancer probability?** Softmax normalizes numerical outputs, but we have not established that predictions with a score of 80% are correct 80% of the time in a clinical population.

**Is this explainable AI?** The site explains the class meaning and scoring process. It does not provide a validated lesion heatmap or a causal explanation of the image. Grad-CAM or occlusion analysis could be added and separately evaluated later.

**Why can't you give Stage I–IV from this JPEG?** Stage depends on tumor extent, lymph nodes and spread, with clinical evaluation. IQ-OTH/NCCD has only category labels. NSCLC-Radiomics has historical stage labels, but its inspected 422-record CSV has no Stage IV examples and requires full imaging linkage for staging research. We have not trained a staging model.

## What to do next

1. Rehearse the seven slides and live demo. Keep downloaded samples ready for unreliable internet.
2. Ask faculty whether three-class CNN classification satisfies the course requirement. Do not claim automatic cancer staging.
3. For improvement, obtain a dataset with reliable patient IDs, use patient-independent splits and evaluate transfer learning on a separate test set.
4. For stage research, define the staging edition and intended classes. Obtain linked full CT studies and clinical labels, resolve ambiguous codes, and include Stage IV cases if that is an intended output. Train and test it as a separate task.
5. For affected-area percentages, obtain validated lung and tumor masks. A segmentation fraction is a different measurement and still does not determine clinical stage.

## Sources

- CT definition: https://www.cancer.gov/about-cancer/diagnosis-staging/ct-scans-fact-sheet
- Dataset: https://data.mendeley.com/datasets/bhmdr45bh2/4
- CNN concepts: https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- Staging dataset: https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
- Staging overview: https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/staging-nsclc.html
