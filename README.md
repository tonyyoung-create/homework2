# Enhanced Spam Email Detection System

This repository is an end-to-end, reproducible spam email detection project inspired by Chapter 3 of the Packt book "Hands-On Artificial Intelligence for Cybersecurity". It provides preprocessing, training, visualizations, a Streamlit UI, and CI-ready tests.

## Project layout

Top-level structure (important folders):

```
.
├── data/               # Dataset (sample or user-provided)
├── models/             # Trained model artifacts (gitignored - add to release)
├── notebooks/          # CRISP-DM Jupyter notebook
├── openspec/           # OpenSpec documentation and proposals
├── src/                # Source code (app, preprocessing, model, train, viz)
└── tests/               # Unit tests
```

## Key features

- CRISP-DM notebook for exploratory analysis and experiment notes
- Robust preprocessing with NLTK fallback (no network required)
- End-to-end training script (`src/train.py`) that saves model artifacts
- Streamlit app (`src/app.py`) for single/batch prediction and performance analysis
- Visualization utilities with graceful fallbacks when optional packages are missing
- Unit tests and GitHub Actions CI

## Quickstart (local)

1. Clone the repo:

```pwsh
git clone https://github.com/tonyyoung-create/homework2.git
cd homework2
```

2. Install dependencies (recommended to use a virtualenv):

```pwsh
python -m pip install -r requirements.txt
```

3. Run tests:

```pwsh
python -m pytest -q
```

4. Run the Streamlit app:

```pwsh
streamlit run src/app.py
```

5. (Optional) Train a model locally and save artifacts:

```pwsh
python src/train.py
# This will create files under `models/` (spam_detector.joblib and spam_detector_vectorizer.joblib)
```

## Notes about deployment

- When deploying to Streamlit Cloud, set the main file to `src/app.py` and ensure `requirements.txt` is up-to-date. The repository currently includes runtime fixes (e.g., `wordcloud`, `Pillow`, `joblib`) to reduce deployment failures.
- Model artifacts are not committed to the repo by default. Add trained artifacts to a GitHub release or to the `models/` directory in your deployment environment.

### Automatic model provisioning (optional)

The app supports downloading model artifacts at startup when they are not present locally. This is useful for Streamlit Cloud deployments where you prefer not to commit large binary files.

1. Upload your artifacts (the two files created by `src/train.py`) somewhere accessible (GitHub release assets, S3, or signed URLs).
2. In Streamlit Cloud (App Settings → Advanced settings), add environment variables:
	- `MODEL_URL` — URL to `spam_detector.joblib`
	- `VECTORIZER_URL` — URL to `spam_detector_vectorizer.joblib`

On app startup the code will attempt to download these files into `models/` and then load them. If download fails or the env vars are not set, the app will show a warning and skip predictions.

Security note: For private artifacts use presigned URLs or a secured storage and set the URLs as Streamlit secrets; avoid committing credentials to the repository.

## Publishing trained artifacts (convenience)

If you trained a model locally and want to publish it as a GitHub Release for Streamlit Cloud to download, there is a convenience script included.

- `scripts/publish_model.ps1` — PowerShell script that uses GitHub CLI `gh` to create a release and upload the two artifacts produced by `src/train.py`.

Prerequisites:
- Install GitHub CLI and login: `gh auth login`

Example (PowerShell):
```pwsh
# from repo root
.\scripts\publish_model.ps1 -ModelPath models\spam_detector.joblib -VectorizerPath models\spam_detector_vectorizer.joblib -Tag v1.0.0 -ReleaseName "v1.0.0 - Spam model"
```

After creating the release, open the release page and copy the asset URLs. Use those URLs as `MODEL_URL` and `VECTORIZER_URL` in Streamlit Cloud (App Settings → Advanced → Environment variables) so the app will download artifacts at startup.

## Contributing and OpenSpec

Follow the OpenSpec workflow documented in the `openspec/` folder for proposing features. Create feature branches from `main`, add tests, and submit pull requests.

## License & Acknowledgements

This project builds on examples from the Packt book "Hands-On Artificial Intelligence for Cybersecurity". See `ACKNOWLEDGEMENTS.md` for attribution and license details.
