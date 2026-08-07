import api from '@/services/api'
import router from '@/router'

const state = () => ({
  token: localStorage.getItem('token'),
  user: null,
  isAuthenticated: !!localStorage.getItem('token')
})

const mutations = {
  SET_TOKEN(state, token) {
    state.token = token
    state.isAuthenticated = !!token
  },
  SET_USER(state, user) {
    state.user = user
  },
  LOGOUT(state) {
    state.token = null
    state.user = null
    state.isAuthenticated = false
  }
}

const actions = {
  async login({ commit }, credentials) {
    try {
      const response = await api.post('/auth/login/', credentials)
      const { token, user } = response.data
      
      localStorage.setItem('token', token)
      commit('SET_TOKEN', token)
      commit('SET_USER', user)
      
      return response
    } catch (error) {
      throw error
    }
  },
  
  async register({ commit }, userData) {
    try {
      const response = await api.post('/auth/register/', userData)
      return response
    } catch (error) {
      throw error
    }
  },
  
  async logout({ commit }) {
    try {
      await api.post('/auth/logout/')
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      localStorage.removeItem('token')
      commit('LOGOUT')
      router.push('/login')
    }
  },
  
  async fetchUser({ commit }) {
    try {
      const response = await api.get('/profile/')
      commit('SET_USER', response.data)
    } catch (error) {
      console.error('Fetch user error:', error)
    }
  }
}

const getters = {
  token: (state) => state.token,
  user: (state) => state.user,
  isAuthenticated: (state) => state.isAuthenticated,
  isAdmin: (state) => state.user?.user_type === 'admin',
  isSupervisor: (state) => state.user?.user_type === 'supervisor',
  isBoursier: (state) => state.user?.user_type === 'boursier'
}

export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
}