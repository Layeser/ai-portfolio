<template>
  <div class="modal-overlay" @click="closeModal">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>Approbation des utilisateurs</h3>
        <button @click="closeModal" class="close-btn">
          <i class="fas fa-times"></i>
        </button>
      </div>
      
      <div class="modal-body">
        <!-- Filtres -->
        <div class="filters-section">
          <div class="filter-row">
            <div class="filter-group">
              <label>Statut</label>
              <select v-model="filters.status">
                <option value="">Tous les statuts</option>
                <option value="pending">En attente</option>
                <option value="approved">Approuvés</option>
                <option value="rejected">Rejetés</option>
              </select>
            </div>
            
            <div class="filter-group">
              <label>Type d'utilisateur</label>
              <select v-model="filters.userType">
                <option value="">Tous les types</option>
                <option value="boursier">Boursier</option>
                <option value="supervisor">Superviseur</option>
                <option value="admin">Administrateur</option>
              </select>
            </div>
            
            <div class="filter-group">
              <label>Recherche</label>
              <input 
                v-model="filters.search" 
                type="text" 
                placeholder="Nom, email..."
              />
            </div>
          </div>
        </div>
        
        <!-- Liste des utilisateurs -->
        <div class="users-section">
          <div class="section-header">
            <h4>Utilisateurs ({{ filteredUsers.length }})</h4>
            <div class="bulk-actions">
              <button @click="selectAll" class="btn-select-all">
                {{ allSelected ? 'Désélectionner tout' : 'Sélectionner tout' }}
              </button>
              <button 
                @click="bulkApprove" 
                class="btn-bulk-approve"
                :disabled="selectedUsers.length === 0"
              >
                Approuver la sélection ({{ selectedUsers.length }})
              </button>
            </div>
          </div>
          
          <div class="users-list">
            <div 
              v-for="user in filteredUsers" 
              :key="user.id" 
              class="user-item"
              :class="{ selected: selectedUsers.includes(user.id) }"
            >
              <div class="user-selection">
                <input 
                  type="checkbox" 
                  :checked="selectedUsers.includes(user.id)"
                  @change="toggleUserSelection(user.id)"
                />
              </div>
              
              <div class="user-info">
                <div class="user-avatar">
                  {{ user.first_name.charAt(0) }}{{ user.last_name.charAt(0) }}
                </div>
                <div class="user-details">
                  <h5>{{ user.first_name }} {{ user.last_name }}</h5>
                  <p>{{ user.email }}</p>
                  <div class="user-meta">
                    <span class="user-type">{{ getUserTypeLabel(user.user_type) }}</span>
                    <span class="user-date">Inscrit le {{ formatDate(user.date_joined) }}</span>
                  </div>
                </div>
              </div>
              
              <div class="user-status">
                <span :class="getStatusClass(user.status)">
                  {{ getStatusLabel(user.status) }}
                </span>
              </div>
              
              <div class="user-actions">
                <button 
                  v-if="user.status === 'pending'"
                  @click="approveUser(user.id)" 
                  class="btn-approve"
                  title="Approuver"
                >
                  <i class="fas fa-check"></i>
                </button>
                <button 
                  v-if="user.status === 'pending'"
                  @click="rejectUser(user.id)" 
                  class="btn-reject"
                  title="Rejeter"
                >
                  <i class="fas fa-times"></i>
                </button>
                <button 
                  @click="viewUserDetails(user.id)" 
                  class="btn-view"
                  title="Voir détails"
                >
                  <i class="fas fa-eye"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Pagination -->
        <div class="pagination-section">
          <div class="pagination-info">
            Affichage de {{ paginationStart + 1 }} à {{ paginationEnd }} sur {{ totalUsers }} utilisateurs
          </div>
          <div class="pagination-controls">
            <button 
              @click="previousPage" 
              :disabled="currentPage === 1"
              class="btn-page"
            >
              <i class="fas fa-chevron-left"></i>
            </button>
            <span class="page-info">{{ currentPage }} / {{ totalPages }}</span>
            <button 
              @click="nextPage" 
              :disabled="currentPage === totalPages"
              class="btn-page"
            >
              <i class="fas fa-chevron-right"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'

const props = defineProps({
  users: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['close', 'user-approved'])

const toast = useToast()

// État local
const filters = ref({
  status: '',
  userType: '',
  search: ''
})

const selectedUsers = ref([])
const currentPage = ref(1)
const itemsPerPage = 10

// Données simulées - à remplacer par des appels API
const users = ref([
  {
    id: 1,
    first_name: 'John',
    last_name: 'Doe',
    email: 'john@example.com',
    user_type: 'boursier',
    status: 'pending',
    date_joined: new Date('2025-08-25')
  },
  {
    id: 2,
    first_name: 'Jane',
    last_name: 'Smith',
    email: 'jane@example.com',
    user_type: 'supervisor',
    status: 'pending',
    date_joined: new Date('2025-08-26')
  },
  // Plus d'utilisateurs...
])

// Computed
const filteredUsers = computed(() => {
  let filtered = users.value

  if (filters.value.status) {
    filtered = filtered.filter(u => u.status === filters.value.status)
  }
  
  if (filters.value.userType) {
    filtered = filtered.filter(u => u.user_type === filters.value.userType)
  }
  
  if (filters.value.search) {
    const search = filters.value.search.toLowerCase()
    filtered = filtered.filter(u => 
      u.first_name.toLowerCase().includes(search) ||
      u.last_name.toLowerCase().includes(search) ||
      u.email.toLowerCase().includes(search)
    )
  }
  
  return filtered
})

const totalUsers = computed(() => filteredUsers.value.length)
const totalPages = computed(() => Math.ceil(totalUsers.value / itemsPerPage))
const paginationStart = computed(() => (currentPage.value - 1) * itemsPerPage)
const paginationEnd = computed(() => Math.min(paginationStart.value + itemsPerPage, totalUsers.value))
const allSelected = computed(() => selectedUsers.value.length === filteredUsers.value.length)

// Méthodes
const closeModal = () => {
  emit('close')
}

const getUserTypeLabel = (userType) => {
  const types = {
    'boursier': 'Boursier',
    'supervisor': 'Superviseur',
    'admin': 'Administrateur'
  }
  return types[userType] || userType
}

const getStatusLabel = (status) => {
  const labels = {
    'pending': 'En attente',
    'approved': 'Approuvé',
    'rejected': 'Rejeté'
  }
  return labels[status] || status
}

const getStatusClass = (status) => {
  const classes = {
    'pending': 'status-pending',
    'approved': 'status-approved',
    'rejected': 'status-rejected'
  }
  return classes[status] || 'status-default'
}

const formatDate = (date) => {
  return new Intl.DateTimeFormat('fr-FR', {
    day: 'numeric',
    month: 'short',
    year: 'numeric'
  }).format(date)
}

const toggleUserSelection = (userId) => {
  const index = selectedUsers.value.indexOf(userId)
  if (index > -1) {
    selectedUsers.value.splice(index, 1)
  } else {
    selectedUsers.value.push(userId)
  }
}

const selectAll = () => {
  if (allSelected.value) {
    selectedUsers.value = []
  } else {
    selectedUsers.value = filteredUsers.value.map(u => u.id)
  }
}

const approveUser = async (userId) => {
  try {
    // Appel API pour approuver l'utilisateur
    toast.success('Utilisateur approuvé avec succès')
    emit('user-approved')
  } catch (error) {
    toast.error('Erreur lors de l\'approbation')
  }
}

const rejectUser = async (userId) => {
  try {
    // Appel API pour rejeter l'utilisateur
    toast.success('Utilisateur rejeté')
    emit('user-approved')
  } catch (error) {
    toast.error('Erreur lors du rejet')
  }
}

const bulkApprove = async () => {
  try {
    // Appel API pour approuver tous les utilisateurs sélectionnés
    toast.success(`${selectedUsers.value.length} utilisateurs approuvés avec succès`)
    selectedUsers.value = []
    emit('user-approved')
  } catch (error) {
    toast.error('Erreur lors de l\'approbation en masse')
  }
}

const viewUserDetails = (userId) => {
  // Navigation vers la page de détails de l'utilisateur
  console.log('Voir détails utilisateur:', userId)
}

const previousPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

// Lifecycle
onMounted(() => {
  // Charger les utilisateurs depuis l'API
})
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.modal-content {
  background: white;
  border-radius: 12px;
  max-width: 900px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 24px 0 24px;
  border-bottom: 1px solid #e1e5e9;
  padding-bottom: 20px;
}

.modal-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 1.5rem;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #7f8c8d;
  cursor: pointer;
  padding: 8px;
  border-radius: 6px;
  transition: all 0.3s ease;
}

.close-btn:hover {
  background: #f8f9fa;
  color: #2c3e50;
}

.modal-body {
  padding: 24px;
}

.filters-section {
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #e1e5e9;
}

.filter-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.filter-group label {
  color: #2c3e50;
  font-weight: 500;
  font-size: 0.9rem;
}

.filter-group select,
.filter-group input {
  padding: 10px 12px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.3s ease;
}

.filter-group select:focus,
.filter-group input:focus {
  outline: none;
  border-color: #667eea;
}

.users-section {
  margin-bottom: 24px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h4 {
  margin: 0;
  color: #2c3e50;
  font-size: 1.2rem;
}

.bulk-actions {
  display: flex;
  gap: 12px;
}

.btn-select-all,
.btn-bulk-approve {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;
}

.btn-select-all {
  background: #f8f9fa;
  color: #6c757d;
  border: 2px solid #e1e5e9;
}

.btn-select-all:hover {
  background: #e9ecef;
  border-color: #adb5bd;
}

.btn-bulk-approve {
  background: #28a745;
  color: white;
}

.btn-bulk-approve:hover:not(:disabled) {
  background: #218838;
}

.btn-bulk-approve:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.users-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.user-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  background: #f8f9fa;
  transition: all 0.3s ease;
}

.user-item:hover {
  border-color: #667eea;
  background: white;
}

.user-item.selected {
  border-color: #667eea;
  background: #f0f4ff;
}

.user-selection {
  flex-shrink: 0;
}

.user-selection input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
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
  flex-shrink: 0;
}

.user-details h5 {
  margin: 0 0 4px 0;
  color: #2c3e50;
  font-size: 1.1rem;
}

.user-details p {
  margin: 0 0 8px 0;
  color: #7f8c8d;
}

.user-meta {
  display: flex;
  gap: 12px;
  align-items: center;
}

.user-type {
  background: #e1e5e9;
  color: #495057;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.user-date {
  color: #95a5a6;
  font-size: 0.8rem;
}

.user-status {
  flex-shrink: 0;
}

.status-pending {
  background: #fff3cd;
  color: #856404;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-approved {
  background: #d4edda;
  color: #155724;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-rejected {
  background: #f8d7da;
  color: #721c24;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-default {
  background: #e2e3e5;
  color: #383d41;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.user-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.btn-approve,
.btn-reject,
.btn-view {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
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

.pagination-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 20px;
  border-top: 1px solid #e1e5e9;
}

.pagination-info {
  color: #7f8c8d;
  font-size: 0.9rem;
}

.pagination-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.btn-page {
  width: 36px;
  height: 36px;
  border: 2px solid #e1e5e9;
  background: white;
  color: #6c757d;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-page:hover:not(:disabled) {
  border-color: #667eea;
  color: #667eea;
}

.btn-page:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  color: #2c3e50;
  font-weight: 500;
  min-width: 60px;
  text-align: center;
}

@media (max-width: 768px) {
  .modal-overlay {
    padding: 10px;
  }
  
  .modal-content {
    max-height: 95vh;
  }
  
  .filter-row {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .section-header {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .bulk-actions {
    justify-content: center;
  }
  
  .user-item {
    flex-direction: column;
    gap: 12px;
    text-align: center;
  }
  
  .user-info {
    flex-direction: column;
    text-align: center;
  }
  
  .user-actions {
    justify-content: center;
  }
  
  .pagination-section {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }
}
</style>
