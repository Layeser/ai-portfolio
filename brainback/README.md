# BrainBack — Full-stack web application

Modern rewrite of a Django monolith into **Django REST Framework** (API) + **Vue 3** (SPA), with optional Docker Compose (PostgreSQL, Redis, nginx).

## Features

- JWT / token-based authentication flows
- User management (register, password reset/change)
- REST API (`/api/...`) + Django admin
- Vue 3 frontend with Pinia store
- Docker Compose for local full-stack runs

## Tech stack

| Layer | Tools |
|-------|--------|
| Backend | Django 4.2, DRF, CORS, PostgreSQL / SQLite |
| Frontend | Vue 3, Vite, Pinia, Axios |
| Ops | Docker, docker-compose, nginx |

## Quick start (local)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env   # edit SECRET_KEY
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

API: `http://localhost:8000/api` · Admin: `http://localhost:8000/admin`

### Frontend

```bash
cd frontend/vue-project
npm install
npm run dev
```

UI: `http://localhost:3000`

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## Security notes

- Never commit `.env`
- `SECRET_KEY` is read from the environment (see `backend/config/settings.py`)
- Docker Compose demo passwords are for **local** use only — change them before any shared deploy

## Project layout

```
brainback/
├── backend/          # Django + DRF
├── frontend/         # Vue 3 app
├── docker-compose.yml
├── .env.example
└── README.md
```

## Why this repo

Demonstrates software engineering beyond notebooks: API design, auth, frontend/backend split, and containerized delivery.
