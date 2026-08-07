# Test de l'API BrainBack

## Test de l'API d'inscription

```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Test",
    "last_name": "User", 
    "email": "test@example.com",
    "user_type": "boursier",
    "boursier_status": "etudiant",
    "password": "testpass123",
    "password_confirm": "testpass123"
  }'
```

**Résultat attendu :**
```json
{
  "message": "User registered successfully. Waiting for approval.",
  "user_id": 2
}
```

## Test de l'API de connexion

```bash
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpass123"
  }'
```

**Résultat attendu :**
```json
{
  "non_field_errors": ["Your account is not yet approved by the administrator."]
}
```

## Test de l'API de profil (après approbation)

```bash
curl -X GET http://localhost:8000/api/profile/ \
  -H "Authorization: Token YOUR_TOKEN_HERE"
```

## Endpoints disponibles

- `POST /api/auth/register/` - Inscription
- `POST /api/auth/login/` - Connexion  
- `POST /api/auth/logout/` - Déconnexion
- `GET /api/profile/` - Profil utilisateur
- `GET /api/opportunities/` - Liste des opportunités
- `GET /api/applications/` - Liste des candidatures
- `GET /api/mentorship-requests/` - Demandes de mentorat

## Accès à l'admin Django

- URL : http://localhost:8000/admin/
- Email : laye@gmail.com
- Mot de passe : laye123

## Test du frontend

- URL : http://localhost:3000/
- Pages disponibles :
  - `/login` - Page de connexion
  - `/register` - Page d'inscription
  - `/dashboard` - Tableau de bord (après connexion)
