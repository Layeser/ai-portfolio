<template>
  <div class="boursier-dashboard">
    <div class="dashboard-header">
      <h1><i class="fas fa-user-graduate me-3"></i>Mon Espace Boursier</h1>
      <p>Bienvenue dans votre espace personnel CPGE Guinée</p>
    </div>

    <!-- Informations personnelles -->
    <div class="profile-section">
      <div class="profile-card">
        <div class="profile-header">
          <div class="profile-avatar">
            {{ user?.first_name?.charAt(0) }}{{ user?.last_name?.charAt(0) }}
          </div>
          <div class="profile-info">
            <h3>{{ user?.first_name }} {{ user?.last_name }}</h3>
            <p>{{ user?.email }}</p>
            <span class="profile-status" :class="getStatusClass(user?.boursier_status)">
              {{ getStatusLabel(user?.boursier_status) }}
            </span>
          </div>
        </div>
        <div class="profile-actions">
          <button @click="editProfile" class="btn-edit">
            <i class="fas fa-edit me-2"></i>Modifier le profil
          </button>
        </div>
      </div>
    </div>

    <!-- Statistiques personnelles -->
    <div class="personal-stats">
      <div class="stat-card">
        <div class="stat-icon">🎓</div>
        <div class="stat-content">
          <h4>Promotion CPGE</h4>
          <p>{{ user?.promotion_cpge || 'Non renseigné' }}</p>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">🏫</div>
        <div class="stat-content">
          <h4>Établissement CPGE</h4>
          <p>{{ user?.cpge_etablissement || 'Non renseigné' }}</p>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">📚</div>
        <div class="stat-content">
          <h4>Filière</h4>
          <p>{{ user?.filiere_cpge || 'Non renseigné' }}</p>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">🌍</div>
        <div class="stat-content">
          <h4>Pays actuel</h4>
          <p>{{ user?.pays_actuel || 'Non renseigné' }}</p>
        </div>
      </div>
    </div>

    <!-- Actions principales -->
    <div class="main-actions">
      <h3>Actions principales</h3>
      <div class="actions-grid">
        <div class="action-card" @click="viewOpportunities">
          <div class="action-icon">💼</div>
          <h4>Opportunités</h4>
          <p>Découvrir les opportunités disponibles</p>
          <span class="action-count">{{ stats.opportunitiesCount }}</span>
        </div>
        
        <div class="action-card" @click="viewMentorship">
          <div class="action-icon">🎓</div>
          <h4>Mentorat</h4>
          <p>Demander un accompagnement</p>
          <span class="action-count">{{ stats.mentorshipCount }}</span>
        </div>
        
        <div class="action-card" @click="viewApplications">
          <div class="action-icon">📝</div>
          <h4>Mes candidatures</h4>
          <p>Suivre vos candidatures</p>
          <span class="action-count">{{ stats.applicationsCount }}</span>
        </div>
        
        <div class="action-card" @click="viewNetwork">
          <div class="action-icon">🤝</div>
          <h4>Réseau</h4>
          <p>Connecter avec d'autres boursiers</p>
          <span class="action-count">{{ stats.networkCount }}</span>
        </div>
      </div>
    </div>

    <!-- Dernières activités -->
    <div class="recent-activities">
      <h3>Dernières activités</h3>
      <div class="activities-list">
        <div v-for="activity in recentActivities" :key="activity.id" class="activity-item">
          <div class="activity-icon" :class="getActivityIconClass(activity.type)">
            {{ getActivityIcon(activity.type) }}
          </div>
          <div class="activity-content">
            <h5>{{ activity.title }}</h5>
            <p>{{ activity.description }}</p>
            <span class="activity-date">{{ formatDate(activity.date) }}</span>
          </div>
          <div class="activity-status" :class="getActivityStatusClass(activity.status)">
            {{ getActivityStatusLabel(activity.status) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Intention de retour -->
    <div class="return-intention">
      <h3>Intention de retour</h3>
      <div class="intention-form">
        <div class="form-group">
          <label>Intention de retour en Guinée</label>
          <select v-model="intention.return" @change="updateIntention">
            <option value="">Sélectionner une option</option>
            <option value="oui">Oui, je souhaite retourner</option>
            <option value="peut_etre">Peut-être, sous certaines conditions</option>
            <option value="non">Non, pas pour le moment</option>
          </select>
        </div>
        
        <div v-if="intention.return === 'oui' || intention.return === 'peut_etre'" class="form-group">
          <label>Conditions de retour</label>
          <textarea 
            v-model="intention.conditions" 
            placeholder="Décrivez vos conditions de retour..."
            rows="3"
            @blur="updateIntention"
          ></textarea>
        </div>
        
        <div v-if="intention.return === 'oui' || intention.return === 'peut_etre'" class="form-group">
          <label>Besoins d'accompagnement</label>
          <textarea 
            v-model="intention.accompagnement" 
            placeholder="Décrivez vos besoins d'accompagnement..."
            rows="3"
            @blur="updateIntention"
          ></textarea>
        </div>
      </div>
    </div>

    <!-- Modales -->
    <ProfileEditModal 
      v-if="showProfileEdit" 
      @close="showProfileEdit = false"
      @profile-updated="handleProfileUpdated"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useToast } from 'vue-toastification'
import { useAuthStore } from '@/stores/auth'
import ProfileEditModal from './modals/ProfileEditModal.vue'

const toast = useToast()
const authStore = useAuthStore()

// État local
const showProfileEdit = ref(false)
const stats = ref({
  opportunitiesCount: 0,
  mentorshipCount: 0,
  applicationsCount: 0,
  networkCount: 0
})

const intention = ref({
  return: '',
  conditions: '',
  accompagnement: ''
})

const recentActivities = ref([])

// Computed
const user = computed(() => authStore.user)

// Méthodes
const loadDashboardData = async () => {
  try {
    // Charger les statistiques personnelles
    stats.value = {
      opportunitiesCount: 12,
      mentorshipCount: 3,
      applicationsCount: 5,
      networkCount: 45
    }
    
    // Charger les activités récentes
    recentActivities.value = [
      {
        id: 1,
        type: 'opportunity',
        title: 'Nouvelle opportunité disponible',
        description: 'Stage chez Google France',
        date: new Date('2025-08-30'),
        status: 'new'
      },
      {
        id: 2,
        type: 'application',
        title: 'Candidature acceptée',
        description: 'Chez Microsoft',
        date: new Date('2025-08-28'),
        status: 'accepted'
      },
      {
        id: 3,
        type: 'mentorship',
        title: 'Demande de mentorat',
        description: 'En attente de réponse',
        date: new Date('2025-08-25'),
        status: 'pending'
      }
    ]
    
    // Charger l'intention de retour
    if (user.value?.intention_retour) {
      intention.value.return = user.value.intention_retour
      intention.value.conditions = user.value.conditions_retour || ''
      intention.value.accompagnement = user.value.besoins_accompagnement || ''
    }
  } catch (error) {
    toast.error('Erreur lors du chargement des données')
  }
}

const getStatusLabel = (status) => {
  const labels = {
    'etudiant': 'Étudiant en cours',
    'diplome': 'Diplômé'
  }
  return labels[status] || status
}

const getStatusClass = (status) => {
  const classes = {
    'etudiant': 'status-etudiant',
    'diplome': 'status-diplome'
  }
  return classes[status] || 'status-default'
}

const getActivityIcon = (type) => {
  const icons = {
    'opportunity': '💼',
    'application': '📝',
    'mentorship': '🎓',
    'network': '🤝'
  }
  return icons[type] || '📌'
}

const getActivityIconClass = (type) => {
  return `activity-icon-${type}`
}

const getActivityStatusLabel = (status) => {
  const labels = {
    'new': 'Nouveau',
    'pending': 'En attente',
    'accepted': 'Accepté',
    'rejected': 'Rejeté'
  }
  return labels[status] || status
}

const getActivityStatusClass = (status) => {
  const classes = {
    'new': 'status-new',
    'pending': 'status-pending',
    'accepted': 'status-accepted',
    'rejected': 'status-rejected'
  }
  return classes[status] || 'status-default'
}

const formatDate = (date) => {
  return new Intl.DateTimeFormat('fr-FR', {
    day: 'numeric',
    month: 'long',
    year: 'numeric'
  }).format(date)
}

const editProfile = () => {
  showProfileEdit.value = true
}

const viewOpportunities = () => {
  // Navigation vers la page des opportunités
  console.log('Voir opportunités')
}

const viewMentorship = () => {
  // Navigation vers la page de mentorat
  console.log('Voir mentorat')
}

const viewApplications = () => {
  // Navigation vers la page des candidatures
  console.log('Voir candidatures')
}

const viewNetwork = () => {
  // Navigation vers la page du réseau
  console.log('Voir réseau')
}

const updateIntention = async () => {
  try {
    // Appel API pour mettre à jour l'intention de retour
    await authStore.updateProfile({
      intention_retour: intention.value.return,
      conditions_retour: intention.value.conditions,
      besoins_accompagnement: intention.value.accompagnement
    })
    
    toast.success('Intention de retour mise à jour')
  } catch (error) {
    toast.error('Erreur lors de la mise à jour')
  }
}

const handleProfileUpdated = () => {
  showProfileEdit.value = false
  loadDashboardData()
}

// Lifecycle
onMounted(() => {
  loadDashboardData()
})
</script>

<style scoped>
.boursier-dashboard {
  padding: 24px;
  max-width: 1200px;
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

.profile-section {
  margin-bottom: 32px;
}

.profile-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 20px;
}

.profile-avatar {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 2rem;
}

.profile-info h3 {
  margin: 0 0 8px 0;
  color: #2c3e50;
  font-size: 1.5rem;
}

.profile-info p {
  margin: 0 0 12px 0;
  color: #7f8c8d;
}

.profile-status {
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
}

.status-etudiant {
  background: #d1ecf1;
  color: #0c5460;
}

.status-diplome {
  background: #d4edda;
  color: #155724;
}

.status-default {
  background: #e2e3e5;
  color: #383d41;
}

.btn-edit {
  background: #667eea;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.btn-edit:hover {
  background: #5a6fd8;
}

.personal-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  font-size: 2rem;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8f9fa;
  border-radius: 12px;
}

.stat-content h4 {
  margin: 0 0 4px 0;
  color: #7f8c8d;
  font-size: 0.9rem;
  font-weight: 500;
}

.stat-content p {
  margin: 0;
  color: #2c3e50;
  font-size: 1.1rem;
  font-weight: 600;
}

.main-actions {
  margin-bottom: 32px;
}

.main-actions h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.3rem;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.action-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  cursor: pointer;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
}

.action-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.action-icon {
  font-size: 2.5rem;
  margin-bottom: 16px;
  display: block;
}

.action-card h4 {
  margin: 0 0 8px 0;
  color: #2c3e50;
  font-size: 1.2rem;
}

.action-card p {
  margin: 0;
  color: #7f8c8d;
  font-size: 0.9rem;
  line-height: 1.4;
}

.action-count {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #667eea;
  color: white;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
}

.recent-activities {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  margin-bottom: 32px;
}

.recent-activities h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.3rem;
}

.activities-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.activity-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  background: #f8f9fa;
}

.activity-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  font-size: 1.2rem;
}

.activity-icon-opportunity {
  background: #e3f2fd;
}

.activity-icon-application {
  background: #e8f5e8;
}

.activity-icon-mentorship {
  background: #fff3e0;
}

.activity-icon-network {
  background: #f3e5f5;
}

.activity-content {
  flex: 1;
}

.activity-content h5 {
  margin: 0 0 4px 0;
  color: #2c3e50;
  font-size: 1rem;
}

.activity-content p {
  margin: 0 0 4px 0;
  color: #7f8c8d;
  font-size: 0.9rem;
}

.activity-date {
  color: #95a5a6;
  font-size: 0.8rem;
}

.activity-status {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-new {
  background: #d1ecf1;
  color: #0c5460;
}

.status-pending {
  background: #fff3cd;
  color: #856404;
}

.status-accepted {
  background: #d4edda;
  color: #155724;
}

.status-rejected {
  background: #f8d7da;
  color: #721c24;
}

.return-intention {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.return-intention h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.3rem;
}

.intention-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-group label {
  color: #2c3e50;
  font-weight: 500;
  font-size: 0.9rem;
}

.form-group select,
.form-group textarea {
  padding: 12px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.3s ease;
}

.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
}

@media (max-width: 768px) {
  .boursier-dashboard {
    padding: 16px;
  }
  
  .profile-card {
    flex-direction: column;
    gap: 20px;
    text-align: center;
  }
  
  .personal-stats {
    grid-template-columns: 1fr;
  }
  
  .actions-grid {
    grid-template-columns: 1fr;
  }
  
  .activity-item {
    flex-direction: column;
    text-align: center;
    gap: 12px;
  }
}
</style>
