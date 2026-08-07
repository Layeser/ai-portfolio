.PHONY: install config apps app scaffold app-scaffold run makemigrations migrate createsuperuser seed_students import-alumni import-alumni-file prod-check create-admin

$PYTHON_VERSION := python3

install:
	pip install -r requirements.txt
	pip install --force-reinstall "Django==5.2.2"

# Usage: make apps name=apps
config: 
	django-admin startproject config .

apps:
	mkdir -p $(name)


# Create a Django app inside apps/
# Usage: make app name=students
app:
	mkdir -p apps/$(name)
	python manage.py startapp $(name) apps/$(name)
	@echo "Created Django app: $(name)"

# Add templates/static folders to an existing app
# Usage: make scaffold name=students
scaffold:
	mkdir -p apps/$(name)/templates/$(name)
	mkdir -p apps/$(name)/static/$(name)/css
	mkdir -p apps/$(name)/static/$(name)/scss
	mkdir -p apps/$(name)/static/$(name)/js
	mkdir -p apps/$(name)/static/$(name)/images
	@echo "Scaffolded static and template folders for: $(name)"

app-scaffold:
	@echo "Scaffolding app: $(name)"
	make app name=$(name)
	make scaffold name=$(name)


templates-static:
	@echo "Creating global templates and static folders"
	mkdir -p templates
	mkdir -p templates/components
	mkdir -p templates/layouts
	mkdir -p static/components/css
	mkdir -p static/components/scss
	mkdir -p static/components/js
	mkdir -p static/components/images

	mkdir -p static/layouts/css
	mkdir -p static/layouts/scss
	mkdir -p static/layouts/js
	mkdir -p static/layouts/images

	mkdir -p templates/home
	mkdir -p static/home/css
	mkdir -p static/home/scss
	mkdir -p static/home/js
	mkdir -p static/home/images


makemigrations:
	$(PYTHON_VERSION) ./manage.py makemigrations

migrate:
	$(PYTHON_VERSION) ./manage.py migrate

migrate-all: makemigrations migrate

PORT ?= 8000

run:
	$(PYTHON_VERSION) ./manage.py runserver 127.0.0.1:$(PORT)

createsuperuser:
	$(PYTHON_VERSION) ./manage.py createsuperuser


collectstatic:
	$(PYTHON_VERSION) ./manage.py collectstatic --noinput

# Makefile
sass:
	@echo "Watching SCSS → CSS (un seul process)…"
	sass --watch \
		static/components/scss:static/components/css \
		static/layouts/scss:static/layouts/css \
		static/home/scss:static/home/css \
		apps/core/static/core/scss:apps/core/static/core/css \
		apps/win/static/win/scss:apps/win/static/win/css \
		--poll

sass-watch:
	sass --watch static:static --poll

seed_students:
	$(PYTHON_VERSION) ./manage.py seed_students --count=100 --overwrite


resetdb:
	@echo "Resetting database and migrations..."
	rm -f db.sqlite3
	# Supprime uniquement les fichiers de migrations des apps du projet, pas ceux de Django
	find apps -type f -path "*/migrations/*.py" -not -name "__init__.py" -delete
	find apps -type f -path "*/migrations/*.pyc" -delete
	$(PYTHON_VERSION) ./manage.py makemigrations
	$(PYTHON_VERSION) ./manage.py migrate
	@echo "Database reset complete."

import-alumni:
	$(PYTHON_VERSION) ./manage.py import_alumni_accounts --file data/data_alumni_compte.xlsx --password password124cpge


# ----------------------
# Remote MySQL helpers
# ----------------------
DB_HOST ?= cpge-database.cov4oeo8ox8y.us-east-1.rds.amazonaws.com
DB_USER ?= admin
DB_NAME ?= cpge_database
DB_PASS ?= Entrepreneur25mere

mysql:
	mysql -h $(DB_HOST) -u $(DB_USER) -p

# Supprime toutes les tables (drop + recreate le schéma vide)
db-drop:
	@echo "Dropping schema $(DB_NAME) on $(DB_HOST)…"
	mysql -h $(DB_HOST) -u $(DB_USER) -p$(DB_PASS) -e "DROP DATABASE IF EXISTS \`$(DB_NAME)\`;"
	@echo "Schema dropped."

# Crée la base si absente avec l'encodage correct
db-create:
	@echo "Creating schema $(DB_NAME) on $(DB_HOST)…"
	mysql -h $(DB_HOST) -u $(DB_USER) -p$(DB_PASS) -e "CREATE DATABASE IF NOT EXISTS \`$(DB_NAME)\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
	@echo "Schema created."

# Vérifie la configuration déploiement (Django check --deploy)
prod-check:
	$(PYTHON_VERSION) ./manage.py check --deploy --fail-level WARNING

create-admin:
	@echo "Création d'un compte administrateur…"
	$(PYTHON_VERSION) ./manage.py create_admin