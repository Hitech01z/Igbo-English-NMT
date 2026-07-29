# Igbo-English Neural Machine Translation System Using Iterative Back Translation

## Project Overview

This project presents an Igbo-English Neural Machine Translation (NMT) system developed using Iterative Back Translation to improve low-resource language translation quality.

The system supports:

* Igbo to English translation
* English to Igbo translation
* Dataset contribution
* Dataset exploration
* Translation history
* Analytics dashboard
* BLEU score evaluation

---

## Objectives

### General Objective

To develop an effective neural machine translation system for translating between Igbo and English languages.

### Specific Objectives

* Build an Igbo-English parallel corpus.
* Implement Iterative Back Translation.
* Evaluate translation quality using BLEU scores.
* Develop a web interface for users.
* Support community dataset contributions.

---

## Technologies Used

### Backend

* FastAPI
* Python 3.11
* Pandas
* Transformers
* PyTorch
* NLTK

### Frontend

* React
* Vite
* TailwindCSS

---

## Installation

### Clone Repository

git clone https://github.com/yourname/Igbo-English-NMT.git

cd Igbo-English-NMT

---

## Backend Setup

cd backend

python -m venv .venv

source .venv/Scripts/activate

pip install -r requirements.txt

uvicorn app.main:app --reload

---

## Frontend Setup

cd frontend

npm install

npm run dev

---

## API Endpoints

GET /

POST /translate

GET /dashboard

GET /dataset

POST /contribute

GET /history

GET /explorer

GET /admin/pending

---

## Dataset Structure

dataset/

education/

health/

agriculture/

market/

culture/

technology/

contributions/

---

## Evaluation

BLEU Score Metric:

BLEU = 31.62 (development version)

---

## Future Work

* Fine-tune M2M100 models.
* Deploy on cloud infrastructure.
* Add speech translation.
* Add voice recognition.
* Develop Android applications.
