<template>
  <div class="dashboard-container">
    <!-- Loading state -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>Chargement du tableau de bord...</p>
    </div>
    
    <!-- Dashboard selon le type d'utilisateur -->
    <div v-else>
      <AdminDashboard v-if="userType === 'admin'" />
      <SupervisorDashboard v-else-if="userType === 'supervisor'" />
      <BoursierDashboard v-else-if="userType === 'boursier'" />
      
      <!-- Fallback pour les autres types d'utilisateurs -->
      <div v-else class="fallback-dashboard">
        <div class="fallback-content">
          <h2>Tableau de bord non disponible</h2>
          <p>Votre type d'utilisateur ne dispose pas encore d'un tableau de bord personnalisé.</p>
          <p>Type d'utilisateur détecté : <strong>{{ userType || 'Non défini' }}</strong></p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import AdminDashboard from './AdminDashboard.vue'
import SupervisorDashboard from './SupervisorDashboard.vue'
import BoursierDashboard from './BoursierDashboard.vue'

const authStore = useAuthStore()
const loading = ref(true)

// Computed
const userType = computed(() => {
  if (!authStore.user) return null
  return authStore.user.user_type
})

// Lifecycle
onMounted(async () => {
  try {
    // S'assurer que le profil utilisateur est chargé
    if (!authStore.user) {
      await authStore.fetchProfile()
    }
    
    // Simuler un délai de chargement pour une meilleure UX
    setTimeout(() => {
      loading.value = false
    }, 500)
  } catch (error) {
    console.error('Erreur lors du chargement du dashboard:', error)
    loading.value = false
  }
})
</script>

<style scoped>
.dashboard-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #e1e5e9;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-container p {
  color: #7f8c8d;
  font-size: 1.1rem;
  margin: 0;
}

.fallback-dashboard {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  padding: 40px;
}

.fallback-content {
  background: white;
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  text-align: center;
  max-width: 500px;
}

.fallback-content h2 {
  color: #2c3e50;
  margin-bottom: 16px;
  font-size: 1.8rem;
}

.fallback-content p {
  color: #7f8c8d;
  margin-bottom: 12px;
  line-height: 1.5;
}

.fallback-content p:last-child {
  margin-bottom: 0;
}

.fallback-content strong {
  color: #667eea;
}
</style>
