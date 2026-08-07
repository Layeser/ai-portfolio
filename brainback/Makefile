.PHONY: help install dev build test clean docker-up docker-down docker-build

help: ## Afficher cette aide
	@echo "BrainBack FS - Commandes disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer toutes les dépendances
	@echo "📦 Installation des dépendances..."
	@cd backend && python -m venv .venv
	@cd backend && .venv/bin/pip install -r requirements.txt
	@cd frontend/vue-project && npm install
	@echo "✅ Installation terminée"

dev: ## Démarrer le projet en mode développement
	@echo "🚀 Démarrage du projet en mode développement..."
	@./start.sh

build: ## Construire le projet pour la production
	@echo "🔨 Construction du projet..."
	@cd frontend/vue-project && npm run build
	@cd backend && python manage.py collectstatic --noinput
	@echo "✅ Construction terminée"

test: ## Lancer les tests
	@echo "🧪 Lancement des tests..."
	@cd backend && python manage.py test
	@cd frontend/vue-project && npm run test
	@echo "✅ Tests terminés"

clean: ## Nettoyer les fichiers temporaires
	@echo "🧹 Nettoyage..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -delete
	@find . -type d -name ".pytest_cache" -delete
	@find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Nettoyage terminé"

docker-up: ## Démarrer les services Docker
	@echo "🐳 Démarrage des services Docker..."
	@docker-compose up -d
	@echo "✅ Services Docker démarrés"

docker-down: ## Arrêter les services Docker
	@echo "🛑 Arrêt des services Docker..."
	@docker-compose down
	@echo "✅ Services Docker arrêtés"

docker-build: ## Construire et démarrer les services Docker
	@echo "🔨 Construction et démarrage des services Docker..."
	@docker-compose up --build -d
	@echo "✅ Services Docker construits et démarrés"

migrate: ## Appliquer les migrations de base de données
	@echo "🗄️ Application des migrations..."
	@cd backend && python manage.py makemigrations
	@cd backend && python manage.py migrate
	@echo "✅ Migrations appliquées"

superuser: ## Créer un superutilisateur
	@echo "👑 Création d'un superutilisateur..."
	@cd backend && python manage.py createsuperuser
	@echo "✅ Superutilisateur créé"

shell: ## Ouvrir un shell Django
	@echo "🐍 Ouverture du shell Django..."
	@cd backend && python manage.py shell

collectstatic: ## Collecter les fichiers statiques
	@echo "📁 Collecte des fichiers statiques..."
	@cd backend && python manage.py collectstatic --noinput
	@echo "✅ Fichiers statiques collectés"

logs: ## Afficher les logs des services
	@echo "📋 Affichage des logs..."
	@docker-compose logs -f

backup: ## Sauvegarder la base de données
	@echo "💾 Sauvegarde de la base de données..."
	@cd backend && python manage.py dumpdata > backup_$(shell date +%Y%m%d_%H%M%S).json
	@echo "✅ Sauvegarde terminée"

restore: ## Restaurer la base de données (usage: make restore FILE=backup_file.json)
	@echo "🔄 Restauration de la base de données..."
	@cd backend && python manage.py loaddata ../$(FILE)
	@echo "✅ Restauration terminée"
