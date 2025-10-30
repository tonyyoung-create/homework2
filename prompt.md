# Project Prompt: Spam Email Detection (CRISP-DM, OpenSpec, Streamlit)

## Project Overview
This project implements a reproducible, end-to-end spam email detection pipeline using Python, scikit-learn, and Streamlit, following the CRISP-DM methodology. The workflow is inspired by Chapter 3 of the Packt book "Hands-On Artificial Intelligence for Cybersecurity" and is designed for transparency, reproducibility, and ease of deployment.

## Key Features
- **CRISP-DM Notebook**: Complete Jupyter notebook covering all CRISP-DM phases (business/data understanding, preparation, modeling, evaluation, deployment, testing, experiment logging).
- **OpenSpec Workflow**: All features, proposals, and documentation tracked in the `openspec/` directory for collaborative, auditable development.
- **Robust Preprocessing**: Handles NLTK and fallback tokenization, with unit tests for reliability.
- **Model Training & Persistence**: End-to-end training script, model artifact documentation, and sample data for reproducibility.
- **Streamlit UI**: Interactive app for single email prediction, batch analysis, and model performance visualization.
- **CI/CD**: GitHub Actions for automated testing and deployment.
- **Licensing & Attribution**: MIT license, with clear acknowledgements to the Packt source material.

## Directory Structure
- `src/`: Core modules (preprocessing, model, visualization, app, training)
- `data/`: Sample dataset for reproducibility
- `models/`: Model artifacts and documentation
- `notebooks/`: CRISP-DM-compliant Jupyter notebook
- `openspec/`: Project documentation, proposals, and workflow
- `tests/`: Unit tests for core modules
- `.github/workflows/`: CI pipeline

## Reproducibility & Deployment
- All code, data, and models are versioned and documented.
- The project is ready for deployment to GitHub and Streamlit Cloud.
- See below for deployment instructions.

## Recent fixes (local workspace)
- Added runtime dependency fixes: `wordcloud`, `Pillow`, and `joblib` were added to `requirements.txt` to ensure Streamlit Cloud and local runs install required packages.
- Hardened `src/visualization.py` with an import guard for `wordcloud` and a bar-chart fallback to avoid import-time crashes.
- Fixed CSV quoting in `data/sample_emails.csv` so `pandas.read_csv` reads `label` and `text` correctly.
- Normalized imports and improved robustness in `src/train.py` and `src/app.py` (safe model/vectorizer checks, better probability formatting, added batch and performance pages).
- Unit tests (`tests/test_preprocessing.py`) run and passed locally.

These changes were committed and pushed to the repository's `main` branch. If you deploy to Streamlit Cloud, it will pick up the latest `requirements.txt` and code on the `main` branch.

---

# GitHub + Streamlit Deployment Steps

## 1. Prepare Repository
- Create a new GitHub repository (e.g., `huanchen1107/2025ML-spamEmail`).
- Push all project files, including code, data, models, notebooks, and documentation.
- Ensure `requirements.txt` includes all dependencies (scikit-learn, pandas, numpy, streamlit, plotly, matplotlib, seaborn, joblib, nltk, wordcloud, pytest, black, mypy, jupyter).
- Add `.github/workflows/python-ci.yml` for CI testing.

## 2. Streamlit Cloud Deployment
- Sign in to [Streamlit Cloud](https://streamlit.io/cloud).
- Click "New app" and connect your GitHub repo.
- Set the main file path to `src/app.py`.
- Configure environment variables/secrets if needed (e.g., for email APIs).
- Click "Deploy". Streamlit Cloud will install dependencies and launch the app.

Notes for a successful deploy:
- Make sure `requirements.txt` is up to date (it now includes `wordcloud`, `Pillow`, `joblib`).
- If your app fails on Streamlit Cloud, open the app logs (Settings → Logs) to see dependency install errors or traceback; common fixes are adding missing packages to `requirements.txt` or increasing the Python version in Streamlit settings.

## 3. Continuous Integration (CI)
- GitHub Actions will run tests on every push via `.github/workflows/python-ci.yml`.
- Ensure all tests pass before merging PRs.

## 4. Maintenance
- Update `requirements.txt` and documentation as needed.
- Use the `openspec/` directory for proposals, feature tracking, and collaborative planning.

Local run & debug commands (PowerShell)
```pwsh
# install deps
python -m pip install -r requirements.txt

# run tests
python -m pytest -q

# run the Streamlit app locally
streamlit run src/app.py
```

---

# References
- Packt Publishing, "Hands-On Artificial Intelligence for Cybersecurity" (MIT License)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

# Acknowledgements
See `ACKNOWLEDGEMENTS.md` for full attributions.
