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

## 3. Continuous Integration (CI)
- GitHub Actions will run tests on every push via `.github/workflows/python-ci.yml`.
- Ensure all tests pass before merging PRs.

## 4. Maintenance
- Update `requirements.txt` and documentation as needed.
- Use the `openspec/` directory for proposals, feature tracking, and collaborative planning.

---

# References
- Packt Publishing, "Hands-On Artificial Intelligence for Cybersecurity" (MIT License)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

# Acknowledgements
See `ACKNOWLEDGEMENTS.md` for full attributions.
