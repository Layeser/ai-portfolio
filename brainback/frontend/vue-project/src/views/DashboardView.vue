<template>
  <div class="dashboard">
    <Navigation />
    
    <header class="dashboard-header">
      <div class="header-content">
        <h1>Tableau de bord</h1>
        <div class="welcome-message">
          <span>Bienvenue, {{ user?.first_name || 'Utilisateur' }}</span>
        </div>
      </div>
    </header>

    <main class="dashboard-main">
      <div class="dashboard-grid">
        <div class="dashboard-card">
          <h3>Profil</h3>
          <p>Gérez vos informations personnelles</p>
          <button class="card-btn">Voir le profil</button>
        </div>

        <div class="dashboard-card">
          <h3>Opportunités</h3>
          <p>Découvrez les opportunités disponibles</p>
          <button class="card-btn">Voir les opportunités</button>
        </div>

        <div class="dashboard-card">
          <h3>Demandes de mentorat</h3>
          <p>Gérez vos demandes de mentorat</p>
          <button class="card-btn">Voir les demandes</button>
        </div>

        <div class="dashboard-card">
          <h3>Candidatures</h3>
          <p>Suivez vos candidatures</p>
          <button class="card-btn">Voir les candidatures</button>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useToast } from 'vue-toastification'
import { useAuthStore } from '@/stores/auth'
import Navigation from '@/components/Navigation.vue'

const router = useRouter()
const toast = useToast()
const authStore = useAuthStore()

const user = computed(() => authStore.user)

onMounted(async () => {
  if (!authStore.isAuthenticated) {
    router.push('/login')
    return
  }
  
  try {
    await authStore.fetchProfile()
  } catch (error) {
    toast.error('Erreur lors du chargement du profil')
    router.push('/login')
  }
})

// La déconnexion est maintenant gérée par le composant Navigation
</script>

<style scoped>
.dashboard {
  min-height: 100vh;
  background-color: #f8f9fa;
}

.dashboard-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px 0;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h1 {
  margin: 0;
  font-size: 28px;
  font-weight: 600;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 20px;
}

.logout-btn {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.logout-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

.dashboard-main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 30px;
}

.dashboard-card {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.dashboard-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.dashboard-card h3 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 20px;
  font-weight: 600;
}

.dashboard-card p {
  margin: 0 0 20px 0;
  color: #666;
  line-height: 1.5;
}

.card-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.card-btn:hover {
  transform: translateY(-2px);
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 15px;
    text-align: center;
  }
  
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}
</style>
