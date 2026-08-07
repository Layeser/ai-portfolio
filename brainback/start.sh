#!/bin/bash

echo "🚀 Démarrage de BrainBack FS..."
echo "================================"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier si Node.js est installé
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier si npm est installé
if ! command -v npm &> /dev/null; then
    echo "❌ npm n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

echo "✅ Vérifications des prérequis terminées"
echo ""

# Démarrer le backend
echo "🐍 Démarrage du backend Django..."
cd backend

# Vérifier si l'environnement virtuel existe
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv .venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source .venv/bin/activate

# Installer les dépendances si nécessaire
if [ ! -f ".venv/pip-installed" ]; then
    echo "📥 Installation des dépendances Python..."
    pip install -r requirements.txt
    touch .venv/pip-installed
fi

# Démarrer le serveur backend en arrière-plan
echo "🚀 Lancement du serveur backend sur http://localhost:8000..."
python manage.py runserver &
BACKEND_PID=$!

# Attendre que le backend démarre
sleep 3

# Démarrer le frontend
echo ""
echo "⚡ Démarrage du frontend Vue.js..."
cd ../frontend/vue-project

# Installer les dépendances si nécessaire
if [ ! -d "node_modules" ]; then
    echo "📥 Installation des dépendances Node.js..."
    npm install
fi

# Démarrer le serveur frontend
echo "🚀 Lancement du serveur frontend sur http://localhost:3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 Projet démarré avec succès !"
echo "================================"
echo "🌐 Frontend: http://localhost:3000 (ou port suivant si occupé)"
echo "🔧 Backend API: http://localhost:8000/api"
echo "⚙️  Admin Django: http://localhost:8000/admin"
echo ""
echo "Pour arrêter les serveurs, appuyez sur Ctrl+C"

# Fonction de nettoyage
cleanup() {
    echo ""
    echo "🛑 Arrêt des serveurs..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Serveurs arrêtés"
    exit 0
}

# Capturer Ctrl+C
trap cleanup SIGINT

# Attendre indéfiniment
wait
