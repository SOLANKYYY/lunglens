# GitHub → Vercel

## Static website deployment

1. Follow the README to push this folder to your own GitHub repository.
2. In Vercel, choose **Add New → Project**, connect GitHub and import that repository.
3. Use the repository root as the project root.
4. Set framework preset to **Other**.
5. No install or build command is needed for this static website. Leave the Build Command empty (enable Override if necessary).
6. Output directory: **web**. The included `vercel.json` declares it.
7. Deploy. Test the published URL on desktop and mobile.

The only published files should be those in `web/`. Do not set the output directory to the whole repository: training data and source files do not need to be public web assets.

The static website supports the original sample simulations, scan previews, stage-learning cards, dataset links and sample downloads. It gracefully shows “Browser demo” when `/api/health` is unavailable.

## Model inference is a separate service

`python -m backend.server` is a local development server bound to 127.0.0.1. Vercel does not keep that local Python process running. The delivered Vercel configuration does not deploy the NumPy, TensorFlow or PyTorch inference engine.

A later production integration should:

- Package trusted model weights with a production Python API, with dependencies and model versions fixed.
- Implement `/api/health` and `/api/predict` with the response contract in `backend/server.py`.
- Use a same-origin authenticated proxy or correctly configured cross-origin HTTPS requests. Update `web/project.js` deliberately if using a different origin.
- Add authentication, request limits, appropriate storage/privacy controls and monitoring before accepting sensitive uploads.
- Preserve the distinction between a research class score and clinical diagnosis or staging.

Do not put backend secrets into frontend JavaScript. Do not make the standard-library development server publicly accessible.

For your faculty demonstration, use the local Python server; it runs both the website and model on one origin with no cloud backend configuration.

## Troubleshooting

- **404 on the Vercel root:** confirm output directory is `web` and it contains `index.html`.
- **No Run local model button:** the frontend cannot reach a ready Python model. Open the URL printed by `python -m backend.server`, not the Vercel URL or a `file://` page.
- **No model loaded:** retrain using the README command or restore `models/demo`; restart the server.
- **Port occupied:** run `python -m backend.server --port 8001` and open http://127.0.0.1:8001.
- **Changes not visible:** stop and restart the Python server, then refresh. It sends `Cache-Control: no-store` locally.

Official Vercel references:
https://vercel.com/docs/project-configuration/vercel-json
https://vercel.com/docs/builds/configure-a-build
