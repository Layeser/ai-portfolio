<template>
  <header class="header">
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <div class="container">
        <!-- Logo -->
        <router-link class="navbar-brand" to="/">
          <img :src="require('@/assets/images/logo.png')" alt="BrainBack Logo" class="logo-img">
        </router-link>
        
        <!-- Mobile Toggle -->
        <button class="navbar-toggler" type="button" @click="toggleMenu">
          <span class="navbar-toggler-icon"></span>
        </button>
        
        <!-- Navigation Menu -->
        <div class="collapse navbar-collapse" :class="{ show: isMenuOpen }" id="mainNavbar">
          <ul class="navbar-nav me-auto">
            <li class="nav-item">
              <router-link class="nav-link" to="/" exact-active-class="active">
                <i class="fas fa-home me-1"></i>
                Accueil
              </router-link>
            </li>
            
            <template v-if="isAuthenticated">
              <!-- Admin Menu -->
              <li v-if="user?.user_type === 'admin'" class="nav-item">
                <router-link class="nav-link" to="/admin/dashboard" active-class="active">
                  <i class="fas fa-tachometer-alt me-1"></i>
                  Mon Espace
                </router-link>
              </li>
              
              <!-- Supervisor Menu -->
              <li v-else-if="user?.user_type === 'supervisor'" class="nav-item">
                <router-link class="nav-link" to="/supervisor/dashboard" active-class="active">
                  <i class="fas fa-chalkboard-teacher me-1"></i>
                  Mon Espace
                </router-link>
              </li>
              
              <!-- Boursier Menu -->
              <li v-else-if="user?.user_type === 'boursier'" class="nav-item">
                <router-link class="nav-link" to="/boursier/dashboard" active-class="active">
                  <i class="fas fa-chart-bar me-1"></i>
                  Mon Espace
                </router-link>
              </li>
              
              <!-- Win App Menu (for non-boursiers) -->
              <template v-if="user?.user_type !== 'boursier'">
                <li class="nav-item">
                  <router-link class="nav-link" to="/win/visuels" active-class="active">
                    <i class="fas fa-chart-bar me-1"></i>
                    Visuels
                  </router-link>
                </li>
                <li class="nav-item">
                  <router-link class="nav-link" to="/win/promotions" active-class="active">
                    <i class="fas fa-layer-group me-1"></i>
                    Promotions
                  </router-link>
                </li>
              </template>
              
              <!-- Contact Menu -->
              <li v-if="user?.user_type === 'boursier' || user?.user_type === 'admin'" class="nav-item">
                <router-link class="nav-link" to="/contact" active-class="active">
                  <i class="fas fa-address-book me-1"></i>
                  Contacts
                </router-link>
              </li>
            </template>
          </ul>
          
          <!-- User Menu -->
          <UserMenu v-if="isAuthenticated" :user="user" />
          
          <!-- Auth Buttons -->
          <div v-else class="navbar-nav ms-auto">
            <router-link class="nav-link" to="/login">
              <i class="fas fa-sign-in-alt me-1"></i>
              Connexion
            </router-link>
            <router-link class="btn btn-primary btn-lg px-3 ms-2" to="/register">
              <i class="fas fa-user-plus me-1"></i>
              Inscription
            </router-link>
          </div>
        </div>
      </div>
    </nav>
  </header>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useStore } from 'vuex'
import UserMenu from '@/components/common/UserMenu.vue'

const store = useStore()
const isMenuOpen = ref(false)

const isAuthenticated = computed(() => store.getters['auth/isAuthenticated'])
const user = computed(() => store.getters['auth/user'])

const toggleMenu = () => {
  isMenuOpen.value = !isMenuOpen.value
}
</script>