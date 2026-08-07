<template>
  <ul class="navbar-nav ms-auto">
    <li class="nav-item dropdown d-none d-lg-block">
      <a class="nav-link dropdown-toggle d-flex align-items-center" href="#" @click.prevent="toggleDropdown">
        <img v-if="user.profile_photo" :src="user.profile_photo" alt="avatar" class="rounded-circle user-avatar me-2">
        <i v-else class="fas fa-user-circle me-2"></i>
        <span>{{ user.first_name || user.email }}</span>
      </a>
      <ul class="dropdown-menu dropdown-menu-end" :class="{ show: isDropdownOpen }">
        <li>
          <router-link class="dropdown-item" to="/dashboard">
            <i class="fas fa-tachometer-alt me-2"></i>
            Mon Dashboard
          </router-link>
        </li>
        <li>
          <router-link class="dropdown-item" to="/profile">
            <i class="fas fa-user-edit me-2"></i>
            Modifier Profil
          </router-link>
        </li>
        <li>
          <router-link class="dropdown-item" to="/change-password">
            <i class="fas fa-key me-2"></i>
            Changer mot de passe
          </router-link>
        </li>
        <li v-if="user.is_mentor">
          <router-link class="dropdown-item" to="/mentorship">
            <i class="fas fa-handshake me-2"></i>
            Gérer mentorats
          </router-link>
        </li>
        <li><hr class="dropdown-divider"></li>
        <li>
          <a class="dropdown-item" href="#" @click="logout">
            <i class="fas fa-sign-out-alt me-2"></i>
            Déconnexion
          </a>
        </li>
      </ul>
    </li>
  </ul>
</template>

<script setup>
import { ref } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'

const props = defineProps({
  user: {
    type: Object,
    required: true
  }
})

const store = useStore()
const router = useRouter()
const isDropdownOpen = ref(false)

const toggleDropdown = () => {
  isDropdownOpen.value = !isDropdownOpen.value
}

const logout = async () => {
  await store.dispatch('auth/logout')
  router.push('/login')
}
</script>