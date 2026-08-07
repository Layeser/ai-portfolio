<template>
  <div id="app">
    <Header v-if="showHeader" />
    <main class="main-content">
      <router-view />
    </main>
    <Footer v-if="showFooter" />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import Header from '@/components/layout/Header.vue'
import Footer from '@/components/layout/Footer.vue'

const route = useRoute()
const authStore = useAuthStore()

// Afficher le header et footer seulement pour les pages authentifiées
const showHeader = computed(() => authStore.isAuthenticated && route.path !== '/login' && route.path !== '/register')
const showFooter = computed(() => authStore.isAuthenticated && route.path !== '/login' && route.path !== '/register')
</script>

<style>
#app {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  margin: 0;
  padding: 0;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: inherit;
}

.main-content {
  min-height: 100vh;
}
</style>