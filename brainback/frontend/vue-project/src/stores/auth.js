import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/services/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token') || null)
  const isAuthenticated = computed(() => !!token.value && !!user.value)

  const login = async (credentials) => {
    try {
      const response = await api.post('/auth/login/', credentials)
      const { user: userData } = response.data
      
      user.value = userData
      token.value = response.data.token || 'session-token'
      
      if (response.data.token) {
        localStorage.setItem('token', response.data.token)
      }
      
      // Set default authorization header
      api.defaults.headers.common['Authorization'] = `Token ${token.value}`
      
      return { success: true, user: userData }
    } catch (error) {
      console.error('Login error:', error)
      throw error
    }
  }

  const register = async (userData) => {
    try {
      const response = await api.post('/auth/register/', userData)
      return { success: true, message: response.data.message }
    } catch (error) {
      console.error('Registration error:', error)
      throw error
    }
  }

  const logout = () => {
    user.value = null
    token.value = null
    localStorage.removeItem('token')
    delete api.defaults.headers.common['Authorization']
  }

  const fetchProfile = async () => {
    try {
      const response = await api.get('/profile/')
      user.value = response.data
      return response.data
    } catch (error) {
      console.error('Profile fetch error:', error)
      throw error
    }
  }

  const updateProfile = async (profileData) => {
    try {
      const response = await api.put('/profile/', profileData)
      user.value = response.data
      return response.data
    } catch (error) {
      console.error('Profile update error:', error)
      throw error
    }
  }

  const changePassword = async (passwordData) => {
    try {
      const response = await api.post('/auth/password/change/', passwordData)
      return { success: true, message: response.data.message }
    } catch (error) {
      console.error('Password change error:', error)
      throw error
    }
  }

  // Initialize auth state on app start
  const initAuth = async () => {
    if (token.value && !user.value) {
      try {
        await fetchProfile()
      } catch (error) {
        logout()
      }
    }
  }

  return {
    user,
    token,
    isAuthenticated,
    login,
    register,
    logout,
    fetchProfile,
    updateProfile,
    changePassword,
    initAuth
  }
})
