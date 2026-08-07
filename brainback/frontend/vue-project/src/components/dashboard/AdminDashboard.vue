<template>
  <div class="admin-dashboard">
    <div class="dashboard-header">
      <h1><i class="fas fa-shield-alt me-3"></i>Dashboard Administrateur</h1>
      <p>Gestion complète de la plateforme CPGE Guinée</p>
    </div>

    <!-- Statistiques clés -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-content">
          <h3>{{ stats.totalUsers }}</h3>
          <p>Utilisateurs totaux</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⏳</div>
        <div class="stat-content">
          <h3>{{ stats.pendingApprovals }}</h3>
          <p>En attente d'approbation</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-content">
          <h3>{{ stats.approvedUsers }}</h3>
          <p>Utilisateurs approuvés</p>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-content">
          <h3>{{ stats.totalBoursiers }}</h3>
          <p>Total boursiers</p>
        </div>
      </div>
    </div>

    <!-- Actions rapides -->
    <div class="quick-actions">
      <h3>Actions rapides</h3>
      <div class="actions-grid">
        <button @click="showUserApproval = true" class="action-btn">
          <span class="action-icon">👁️</span>
          <span>Approuver utilisateurs</span>
        </button>
        <button @click="showStatistics = true" class="action-btn">
          <span class="action-icon">📈</span>
          <span>Voir statistiques</span>
        </button>
        <button @click="showUserManagement = true" class="action-btn">
          <span class="action-icon">⚙️</span>
          <span>Gérer utilisateurs</span>
        </button>
        <button @click="exportData" class="action-btn">
          <span class="action-icon">📥</span>
          <span>Exporter données</span>
        </button>
      </div>
    </div>

    <!-- Demandes en attente -->
    <div class="pending-requests">
      <h3>Demandes en attente d'approbation</h3>
      <div class="requests-table">
        <div v-for="user in pendingUsers" :key="user.id" class="request-item">
          <div class="user-info">
            <div class="user-avatar">{{ user.first_name.charAt(0) }}{{ user.last_name.charAt(0) }}</div>
            <div class="user-details">
              <h4>{{ user.first_name }} {{ user.last_name }}</h4>
              <p>{{ user.email }}</p>
              <span class="user-type">{{ getUserTypeLabel(user.user_type) }}</span>
            </div>
          </div>
          <div class="request-actions">
            <button @click="approveUser(user.id)" class="btn-approve">Approuver</button>
            <button @click="rejectUser(user.id)" class="btn-reject">Rejeter</button>
            <button @click="viewUserDetails(user.id)" class="btn-view">Voir détails</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Modales -->
    <UserApprovalModal 
      v-if="showUserApproval" 
      @close="showUserApproval = false"
      @user-approved="handleUserApproved"
    />
    
    <StatisticsModal 
      v-if="showStatistics" 
      @close="showStatistics = false"
    />
    
    <UserManagementModal 
      v-if="showUserManagement" 
      @close="showUserManagement = false"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useToast } from 'vue-toastification'
import UserApprovalModal from './modals/UserApprovalModal.vue'
import StatisticsModal from './modals/StatisticsModal.vue'
import UserManagementModal from './modals/UserManagementModal.vue'

const toast = useToast()

// État local
const showUserApproval = ref(false)
const showStatistics = ref(false)
const showUserManagement = ref(false)
const pendingUsers = ref([])
const stats = ref({
  totalUsers: 0,
  pendingApprovals: 0,
  approvedUsers: 0,
  totalBoursiers: 0
})

// Computed
const getUserTypeLabel = (userType) => {
  const types = {
    'boursier': 'Boursier',
    'supervisor': 'Superviseur',
    'admin': 'Administrateur'
  }
  return types[userType] || 'Utilisateur'
}

// Méthodes
const loadDashboardData = async () => {
  try {
    // Charger les statistiques
    // Charger les utilisateurs en attente
    // Ces appels API seront implémentés plus tard
    stats.value = {
      totalUsers: 150,
      pendingApprovals: 12,
      approvedUsers: 138,
      totalBoursiers: 120
    }
    
    pendingUsers.value = [
      {
        id: 1,
        first_name: 'John',
        last_name: 'Doe',
        email: 'john@example.com',
        user_type: 'boursier'
      },
      // Plus d'utilisateurs...
    ]
  } catch (error) {
    toast.error('Erreur lors du chargement des données')
  }
}

const approveUser = async (userId) => {
  try {
    // Appel API pour approuver l'utilisateur
    toast.success('Utilisateur approuvé avec succès')
    await loadDashboardData() // Recharger les données
  } catch (error) {
    toast.error('Erreur lors de l\'approbation')
  }
}

const rejectUser = async (userId) => {
  try {
    // Appel API pour rejeter l'utilisateur
    toast.success('Utilisateur rejeté')
    await loadDashboardData() // Recharger les données
  } catch (error) {
    toast.error('Erreur lors du rejet')
  }
}

const viewUserDetails = (userId) => {
  // Navigation vers la page de détails de l'utilisateur
  console.log('Voir détails utilisateur:', userId)
}

const exportData = async () => {
  try {
    // Appel API pour exporter les données
    toast.success('Export des données lancé')
  } catch (error) {
    toast.error('Erreur lors de l\'export')
  }
}

const handleUserApproved = () => {
  showUserApproval.value = false
  loadDashboardData()
}

// Lifecycle
onMounted(() => {
  loadDashboardData()
})
</script>

<style scoped>
.admin-dashboard {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.dashboard-header {
  text-align: center;
  margin-bottom: 32px;
}

.dashboard-header h1 {
  color: #2c3e50;
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.dashboard-header p {
  color: #7f8c8d;
  font-size: 1.1rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  align-items: center;
  gap: 16px;
  transition: transform 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.stat-icon {
  font-size: 2.5rem;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  color: white;
}

.stat-content h3 {
  font-size: 2rem;
  font-weight: 700;
  color: #2c3e50;
  margin: 0 0 4px 0;
}

.stat-content p {
  color: #7f8c8d;
  margin: 0;
  font-size: 0.9rem;
}

.quick-actions {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  margin-bottom: 32px;
}

.quick-actions h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.3rem;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 20px;
  border: 2px solid #e1e5e9;
  border-radius: 12px;
  background: white;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
}

.action-btn:hover {
  border-color: #667eea;
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
}

.action-icon {
  font-size: 2rem;
}

.pending-requests {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.pending-requests h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.3rem;
}

.requests-table {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.request-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  background: #f8f9fa;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-avatar {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 1.1rem;
}

.user-details h4 {
  margin: 0 0 4px 0;
  color: #2c3e50;
}

.user-details p {
  margin: 0 0 8px 0;
  color: #7f8c8d;
}

.user-type {
  background: #e1e5e9;
  color: #495057;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.request-actions {
  display: flex;
  gap: 12px;
}

.btn-approve, .btn-reject, .btn-view {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
}

.btn-approve {
  background: #28a745;
  color: white;
}

.btn-approve:hover {
  background: #218838;
}

.btn-reject {
  background: #dc3545;
  color: white;
}

.btn-reject:hover {
  background: #c82333;
}

.btn-view {
  background: #17a2b8;
  color: white;
}

.btn-view:hover {
  background: #138496;
}

@media (max-width: 768px) {
  .admin-dashboard {
    padding: 16px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .actions-grid {
    grid-template-columns: 1fr;
  }
  
  .request-item {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }
  
  .request-actions {
    justify-content: center;
  }
}
</style>
