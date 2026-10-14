# Verification record

## Completed in the build environment

- Generated 180 synthetic images and train/validation/test manifest.
- Trained the NumPy linear classifier for 200 epochs with seed 42.
- Selected a checkpoint using validation loss and evaluated the test set: 34/36 correct (94.44%), synthetic data only.
- Six Python tests passed: split sizes, finite inference/no stage, malformed image rejection, patient overlap rejection, duplicate image rejection, and an HTTP integration test.
- The HTTP test served the website and checked model health, valid image upload, invalid image/type, denied cross-origin request and missing-model behavior.
- JavaScript syntax checks passed for app.js and project.js.
- HTML IDs, navigation targets, JavaScript element references, linked assets and Vercel output configuration checked.

## Not verified here

- TensorFlow/PyTorch installation or training execution: those runtimes were not installed. Their Python source compiles; run the supplied commands locally before claiming framework results.
- Browser rendering, interaction, mobile appearance or accessibility with an actual browser. Responsive CSS is included; manual browser QA is still needed.
- A real-data cancer classifier: real CT data was not downloaded or trained.
- Vercel publication or an online Python inference service: neither was deployed.

These limitations are part of the academic handoff. No clinical evaluation or cancer staging validation has been performed.
