<template>
  <nav class="navigation">
    <div class="nav-container">
      <div class="nav-brand">
        <h2>BrainBack</h2>
      </div>
      
      <div class="nav-menu">
        <router-link to="/dashboard" class="nav-link" active-class="active">
          <span class="nav-icon">🏠</span>
          Tableau de bord
        </router-link>
        
        <router-link to="/profile" class="nav-link" active-class="active">
          <span class="nav-icon">👤</span>
          Profil
        </router-link>
        
        <router-link to="/opportunities" class="nav-link" active-class="active">
          <span class="nav-icon">💼</span>
          Opportunités
        </router-link>
        
        <router-link to="/mentorship" class="nav-link" active-class="active">
          <span class="nav-icon">🎓</span>
          Mentorat
        </router-link>
        
        <router-link to="/applications" class="nav-link" active-class="active">
          <span class="nav-icon">📝</span>
          Candidatures
        </router-link>
      </div>
      
      <div class="nav-user">
        <div class="user-info">
          <span class="user-name">{{ user?.first_name || 'Utilisateur' }}</span>
          <span class="user-type">{{ getUserTypeLabel(user?.user_type) }}</span>
        </div>
        <button @click="handleLogout" class="logout-btn">
          <span class="nav-icon">🚪</span>
          Déconnexion
        </button>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useToast } from 'vue-toastification'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const toast = useToast()
const authStore = useAuthStore()

const user = computed(() => authStore.user)

const getUserTypeLabel = (userType) => {
  const types = {
    'boursier': 'Boursier',
    'supervisor': 'Superviseur',
    'admin': 'Administrateur'
  }
  return types[userType] || 'Utilisateur'
}

const handleLogout = () => {
  authStore.logout()
  toast.success('Déconnexion réussie')
  router.push('/login')
}
</script>

<style scoped>
.navigation {
  background: white;
  border-bottom: 1px solid #e1e5e9;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 70px;
}

.nav-brand h2 {
  margin: 0;
  color: #667eea;
  font-size: 24px;
  font-weight: 700;
}

.nav-menu {
  display: flex;
  gap: 30px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: #666;
  font-weight: 500;
  padding: 10px 16px;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.nav-link:hover {
  color: #667eea;
  background: #f8f9fa;
}

.nav-link.active {
  color: #667eea;
  background: #f0f2ff;
}

.nav-icon {
  font-size: 18px;
}

.nav-user {
  display: flex;
  align-items: center;
  gap: 20px;
}

.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.user-name {
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.user-type {
  font-size: 12px;
  color: #666;
}

.logout-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #f8f9fa;
  border: 1px solid #e1e5e9;
  color: #666;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
}

.logout-btn:hover {
  background: #e9ecef;
  color: #495057;
}

@media (max-width: 768px) {
  .nav-container {
    flex-direction: column;
    height: auto;
    padding: 15px 20px;
    gap: 15px;
  }
  
  .nav-menu {
    gap: 15px;
    flex-wrap: wrap;
    justify-content: center;
  }
  
  .nav-user {
    flex-direction: column;
    gap: 10px;
    align-items: center;
  }
  
  .user-info {
    align-items: center;
  }
}
</style>
