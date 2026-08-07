<template>
  <header class="header">
    <div class="header-container">
      <div class="header-brand">
        <router-link to="/" class="brand-link">
          <h1>BrainBack</h1>
        </router-link>
      </div>
      
      <nav class="header-nav">
        <router-link to="/dashboard" class="nav-link">Tableau de bord</router-link>
        <router-link to="/opportunities" class="nav-link">Opportunités</router-link>
        <router-link to="/mentorship" class="nav-link">Mentorat</router-link>
      </nav>
      
      <div class="header-actions">
        <router-link to="/profile" class="profile-link">
          <span class="user-avatar">👤</span>
          <span class="user-name">{{ user?.first_name || 'Utilisateur' }}</span>
        </router-link>
        <button @click="handleLogout" class="logout-btn">Déconnexion</button>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const user = computed(() => authStore.user)

const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}
</script>

<style scoped>
.header {
  background: white;
  border-bottom: 1px solid #e1e5e9;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 70px;
}

.header-brand .brand-link {
  text-decoration: none;
  color: inherit;
}

.header-brand h1 {
  margin: 0;
  color: #667eea;
  font-size: 24px;
  font-weight: 700;
}

.header-nav {
  display: flex;
  gap: 30px;
}

.nav-link {
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

.header-actions {
  display: flex;
  align-items: center;
  gap: 20px;
}

.profile-link {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: #333;
  font-weight: 500;
}

.user-avatar {
  font-size: 20px;
}

.user-name {
  font-size: 14px;
}

.logout-btn {
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
  .header-container {
    flex-direction: column;
    height: auto;
    padding: 15px 20px;
    gap: 15px;
  }
  
  .header-nav {
    gap: 15px;
    flex-wrap: wrap;
    justify-content: center;
  }
  
  .header-actions {
    flex-direction: column;
    gap: 10px;
    align-items: center;
  }
}
</style>
