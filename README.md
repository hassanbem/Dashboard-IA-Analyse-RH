
# Dashboard IA Analyse RH

![Build Status](https://github.com/hassanbem/Dashboard-IA-Analyse-RH/actions/workflows/python-app.yml/badge.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/hassanbem/Dashboard-IA-Analyse-RH)
![GitHub last commit](https://img.shields.io/github/last-commit/hassanbem/Dashboard-IA-Analyse-RH)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

A professional, end-to-end dashboard for HR analytics using AI and data science. This project features a Python backend and a Streamlit frontend for interactive data exploration, sentiment analysis, topic extraction, clustering, and KPI calculation.

![demo](./frontend_backup/demo.png)

## Demo Data

A sample HR analytics dataset is included for demo purposes:
[`backend_backup/data/evaluation_formation_100.csv`](backend_backup/data/evaluation_formation_100.csv)

## Features
- Sentiment analysis with BERT
- Topic extraction and clustering
- KPI calculation and monitoring
- Data anonymization and batch import
- Interactive Streamlit dashboard
- Dockerized backend and frontend

## Quick Start

### Backend
```bash
cd backend_backup
python -m venv .venv  # if not already created
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend_backup
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Folder Structure
- `backend_backup/` — Python backend (APIs, models, services)
- `frontend_backup/` — Streamlit frontend
- `code/` — Model cache and artifacts

## Requirements
- Python 3.8+
- pip
- Streamlit
- Docker (optional)

## License
MIT License (see LICENSE)

## Author
[hassanbem](https://github.com/hassanbem)

---

> This project is designed for HR analytics internships and showcases skills in Python, AI, and dashboard development.
