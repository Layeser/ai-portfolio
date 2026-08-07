<template>
  <div class="register-form">
    <div class="form-container">
      <h2 class="form-title">Inscription</h2>
      
      <form @submit.prevent="handleSubmit" class="form">
        <div class="form-row">
          <div class="form-group">
            <label for="first_name" class="form-label">Prénom</label>
            <input
              id="first_name"
              v-model="form.first_name"
              type="text"
              class="form-input"
              :class="{ 'error': errors.first_name }"
              placeholder="Votre prénom"
              required
            />
            <span v-if="errors.first_name" class="error-message">{{ errors.first_name }}</span>
          </div>

          <div class="form-group">
            <label for="last_name" class="form-label">Nom</label>
            <input
              id="last_name"
              v-model="form.last_name"
              type="text"
              class="form-input"
              :class="{ 'error': errors.last_name }"
              placeholder="Votre nom"
              required
            />
            <span v-if="errors.last_name" class="error-message">{{ errors.last_name }}</span>
          </div>
        </div>

        <div class="form-group">
          <label for="email" class="form-label">Email</label>
          <input
            id="email"
            v-model="form.email"
            type="email"
            class="form-input"
            :class="{ 'error': errors.email }"
            placeholder="votre@email.com"
            required
          />
          <span v-if="errors.email" class="error-message">{{ errors.email }}</span>
        </div>

        <div class="form-group">
          <label for="user_type" class="form-label">Type d'utilisateur</label>
          <select
            id="user_type"
            v-model="form.user_type"
            class="form-input"
            :class="{ 'error': errors.user_type }"
            required
          >
            <option value="">Sélectionnez un type</option>
            <option value="boursier">Boursier</option>
            <option value="supervisor">Superviseur</option>
            <option value="admin">Administrateur</option>
          </select>
          <span v-if="errors.user_type" class="error-message">{{ errors.user_type }}</span>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="password" class="form-label">Mot de passe</label>
            <input
              id="password"
              v-model="form.password"
              type="password"
              class="form-input"
              :class="{ 'error': errors.password }"
              placeholder="Votre mot de passe"
              required
            />
            <span v-if="errors.password" class="error-message">{{ errors.password }}</span>
          </div>

          <div class="form-group">
            <label for="password_confirm" class="form-label">Confirmer le mot de passe</label>
            <input
              id="password_confirm"
              v-model="form.password_confirm"
              type="password"
              class="form-input"
              :class="{ 'error': errors.password_confirm }"
              placeholder="Confirmez votre mot de passe"
              required
            />
            <span v-if="errors.password_confirm" class="error-message">{{ errors.password_confirm }}</span>
          </div>
        </div>

        <button 
          type="submit" 
          class="submit-btn"
          :disabled="loading"
        >
          <span v-if="loading">Inscription...</span>
          <span v-else>S'inscrire</span>
        </button>
      </form>

      <div class="form-footer">
        <p>
          Déjà un compte ? 
          <router-link to="/login" class="link">Se connecter</router-link>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useToast } from 'vue-toastification'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const toast = useToast()
const authStore = useAuthStore()

const loading = ref(false)
const errors = reactive({})

const form = reactive({
  first_name: '',
  last_name: '',
  email: '',
  user_type: '',
  password: '',
  password_confirm: ''
})

const handleSubmit = async () => {
  loading.value = true
  errors.value = {}

  try {
    await authStore.register(form)
    toast.success('Inscription réussie ! Votre compte est en attente d\'approbation.')
    router.push('/login')
  } catch (error) {
    if (error.response?.data) {
      const errorData = error.response.data
      
      if (typeof errorData === 'string') {
        toast.error(errorData)
      } else {
        // Handle field-specific errors
        Object.keys(errorData).forEach(key => {
          if (Array.isArray(errorData[key])) {
            errors[key] = errorData[key][0]
          } else {
            errors[key] = errorData[key]
          }
        })
      }
    } else {
      toast.error('Erreur d\'inscription. Veuillez réessayer.')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.register-form {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.form-container {
  background: white;
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 500px;
}

.form-title {
  text-align: center;
  margin-bottom: 30px;
  color: #333;
  font-size: 28px;
  font-weight: 600;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-label {
  font-weight: 500;
  color: #555;
  font-size: 14px;
}

.form-input {
  padding: 12px 16px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  transition: border-color 0.3s ease;
}

.form-input:focus {
  outline: none;
  border-color: #667eea;
}

.form-input.error {
  border-color: #e74c3c;
}

.error-message {
  color: #e74c3c;
  font-size: 12px;
  margin-top: 4px;
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 14px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.submit-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.form-footer {
  margin-top: 30px;
  text-align: center;
}

.form-footer p {
  margin: 0;
  color: #666;
}

.link {
  color: #667eea;
  text-decoration: none;
  font-weight: 500;
}

.link:hover {
  text-decoration: underline;
}

@media (max-width: 600px) {
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .form-container {
    padding: 20px;
  }
}
</style>
