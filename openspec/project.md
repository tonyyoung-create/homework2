# Project Overview

## Project Name
Enhanced Spam Email Detection System

## Tech Stack
- **Language**: Python 3.11+ (tested on 3.13 in local environment)
- **ML Framework**: scikit-learn (primary), TensorFlow (optional)
- **Visualization**: Plotly, Matplotlib, Streamlit
- **Testing**: pytest, mypy (type checks)
- **Development Tools**: Jupyter Notebooks, VS Code

## Project Goals
1. Enhance email preprocessing capabilities
2. Implement rich visualization components
3. Create user-friendly interfaces (CLI + Streamlit)
4. Improve model metrics and analysis tools

## Acknowledgements
This project builds on and extends example code from the Packt book
"Hands-On Artificial Intelligence for Cybersecurity" and the associated
GitHub repository (PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity).
The Packt code is distributed under the MIT License; please see
`ACKNOWLEDGEMENTS.md` in the project root for the license summary and
attribution guidance.

## Conventions

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Document functions using Google-style docstrings

### File Organization
- Keep notebooks in `notebooks/` directory
- Store models in `models/` directory
- Place datasets in `data/` directory
- Implement source code in `src/` directory
- Write tests in `tests/` directory

### Git Workflow
1. Create feature branches from `main`
2. Submit OpenSpec proposals for new features
3. Follow conventional commits

### Documentation
- Maintain OpenSpec documentation in `openspec/`
- Update README.md with new features
- Include docstrings in all functions
- Document visualization outputs

## Development Process
1. Create OpenSpec proposal for new features
2. Implement features in feature branches
3. Add tests for new functionality
4. Create visualizations for analysis
5. Document changes and update OpenSpec
6. Submit pull request for review

## Dependencies
Core dependencies are managed in `requirements.txt`:
Core dependencies are managed in `requirements.txt`. Recent runtime additions were made to reduce deploy failures; ensure `requirements.txt` includes:

- scikit-learn
- pandas
- numpy
- streamlit
- plotly
- joblib
- wordcloud
- Pillow
- pytest
- black
- mypy

Notes:
- The project was tested locally with Python 3.13; for CI and Streamlit Cloud, Python 3.11+ is recommended unless your deployment target supports newer versions.
- The repository includes a CRISP-DM notebook (`notebooks/01-crisp-dm-spam.ipynb`) and a `src/train.py` script to produce model artifacts.