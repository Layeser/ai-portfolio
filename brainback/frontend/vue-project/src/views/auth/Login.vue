<template>
  <div class="auth-container">
    <div class="auth-card">
      <div class="auth-header">
        <h2>Connexion</h2>
        <p>Connectez-vous à votre compte BrainBack</p>
      </div>
      
      <form @submit.prevent="handleLogin">
        <div class="mb-3">
          <label for="email" class="form-label">Email</label>
          <input
            type="email"
            class="form-control"
            id="email"
            v-model="form.email"
            required
            :class="{ 'is-invalid': errors.email }"
          >
          <div v-if="errors.email" class="invalid-feedback">
            {{ errors.email }}
          </div>
        </div>
        
        <div class="mb-3">
          <label for="password" class="form-label">Mot de passe</label>
          <input
            type="password"
            class="form-control"
            id="password"
            v-model="form.password"
            required
            :class="{ 'is-invalid': errors.password }"
          >
          <div v-if="errors.password" class="invalid-feedback">
            {{ errors.password }}
          </div>
        </div>
        
        <button type="submit" class="btn btn-primary w-100" :disabled="loading">
          <span v-if="loading" class="spinner-border spinner-border-sm me-2"></span>
          Se connecter
        </button>
        
        <div class="auth-links mt-3">
          <router-link to="/forgot-password">Mot de passe oublié ?</router-link>
          <router-link to="/register">Créer un compte</router-link>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'

const store = useStore()
const router = useRouter()

const form = reactive({
  email: '',
  password: ''
})

const errors = ref({})
const loading = ref(false)

const handleLogin = async () => {
  loading.value = true
  errors.value = {}

  try {
    await store.dispatch('auth/login', form)
    router.push('/dashboard')
  } catch (error) {
    if (error.response?.data) {
      errors.value = error.response.data
    } else {
      errors.value = { general: 'Une erreur est survenue' }
    }
  } finally {
    loading.value = false
  }
}
</script>