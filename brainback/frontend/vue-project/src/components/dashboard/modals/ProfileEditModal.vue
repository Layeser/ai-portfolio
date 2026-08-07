<template>
  <div class="modal-overlay" @click="closeModal">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>Modifier le profil</h3>
        <button @click="closeModal" class="close-btn">
          <i class="fas fa-times"></i>
        </button>
      </div>
      
      <form @submit.prevent="handleSubmit" class="modal-body">
        <div class="form-row">
          <div class="form-group">
            <label for="first_name">Prénom *</label>
            <input 
              id="first_name"
              v-model="form.first_name" 
              type="text" 
              required
              placeholder="Votre prénom"
            />
          </div>
          
          <div class="form-group">
            <label for="last_name">Nom *</label>
            <input 
              id="last_name"
              v-model="form.last_name" 
              type="text" 
              required
              placeholder="Votre nom"
            />
          </div>
        </div>
        
        <div class="form-group">
          <label for="email">Email *</label>
          <input 
            id="email"
            v-model="form.email" 
            type="email" 
            required
            placeholder="votre.email@example.com"
          />
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label for="promotion_cpge">Promotion CPGE</label>
            <select id="promotion_cpge" v-model="form.promotion_cpge">
              <option value="">Sélectionner une promotion</option>
              <option v-for="year in promoYears" :key="year" :value="year">{{ year }}</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="filiere_cpge">Filière CPGE</label>
            <select id="filiere_cpge" v-model="form.filiere_cpge">
              <option value="">Sélectionner une filière</option>
              <option v-for="filiere in filieres" :key="filiere" :value="filiere">{{ filiere }}</option>
            </select>
          </div>
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label for="cpge_etablissement">Établissement CPGE</label>
            <input 
              id="cpge_etablissement"
              v-model="form.cpge_etablissement" 
              type="text" 
              placeholder="Nom de l'établissement"
            />
          </div>
          
          <div class="form-group">
            <label for="pays_cpge">Pays CPGE</label>
            <select id="pays_cpge" v-model="form.pays_cpge">
              <option value="">Sélectionner un pays</option>
              <option v-for="pays in paysList" :key="pays" :value="pays">{{ pays }}</option>
            </select>
          </div>
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label for="etablissement_etudes">Établissement d'études actuel</label>
            <input 
              id="etablissement_etudes"
              v-model="form.etablissement_etudes" 
              type="text" 
              placeholder="Nom de l'établissement"
            />
          </div>
          
          <div class="form-group">
            <label for="pays_actuel">Pays actuel</label>
            <select id="pays_actuel" v-model="form.pays_actuel">
              <option value="">Sélectionner un pays</option>
              <option v-for="pays in paysList" :key="pays" :value="pays">{{ pays }}</option>
            </select>
          </div>
        </div>
        
        <div class="form-group">
          <label for="secteur_activite">Secteur d'activité</label>
          <select id="secteur_activite" v-model="form.secteur_activite">
            <option value="">Sélectionner un secteur</option>
            <option v-for="secteur in secteurs" :key="secteur" :value="secteur">{{ secteur }}</option>
          </select>
        </div>
        
        <div class="form-group">
          <label for="bio">Biographie</label>
          <textarea 
            id="bio"
            v-model="form.bio" 
            rows="4"
            placeholder="Parlez-nous de vous, de vos expériences, de vos projets..."
          ></textarea>
        </div>
        
        <div class="form-actions">
          <button type="button" @click="closeModal" class="btn-cancel">
            Annuler
          </button>
          <button type="submit" class="btn-save" :disabled="loading">
            <span v-if="loading" class="loading-spinner-small"></span>
            {{ loading ? 'Sauvegarde...' : 'Sauvegarder' }}
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import { useAuthStore } from '@/stores/auth'

const props = defineProps({
  user: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['close', 'profile-updated'])

const toast = useToast()
const authStore = useAuthStore()

// État local
const loading = ref(false)
const form = ref({
  first_name: '',
  last_name: '',
  email: '',
  promotion_cpge: '',
  filiere_cpge: '',
  cpge_etablissement: '',
  pays_cpge: '',
  etablissement_etudes: '',
  pays_actuel: '',
  secteur_activite: '',
  bio: ''
})

// Données statiques
const promoYears = ['2020', '2021', '2022', '2023', '2024', '2025']
const filieres = ['MP', 'PC', 'PSI', 'TSI', 'BCPST', 'T', 'ATS']
const paysList = [
  'Guinée', 'France', 'Canada', 'États-Unis', 'Allemagne', 'Suisse', 
  'Belgique', 'Luxembourg', 'Royaume-Uni', 'Espagne', 'Italie', 'Pays-Bas'
]
const secteurs = [
  'Informatique', 'Génie Civil', 'Génie Mécanique', 'Génie Électrique',
  'Finance et comptabilité', 'Marketing', 'Ressources humaines', 'Logistique',
  'Santé', 'Enseignement', 'Recherche', 'Consulting', 'Autre'
]

// Méthodes
const closeModal = () => {
  emit('close')
}

const initializeForm = () => {
  if (props.user) {
    form.value = {
      first_name: props.user.first_name || '',
      last_name: props.user.last_name || '',
      email: props.user.email || '',
      promotion_cpge: props.user.promotion_cpge || '',
      filiere_cpge: props.user.filiere_cpge || '',
      cpge_etablissement: props.user.cpge_etablissement || '',
      pays_cpge: props.user.pays_cpge || '',
      etablissement_etudes: props.user.etablissement_etudes || '',
      pays_actuel: props.user.pays_actuel || '',
      secteur_activite: props.user.secteur_activite || '',
      bio: props.user.bio || ''
    }
  }
}

const handleSubmit = async () => {
  try {
    loading.value = true
    
    // Appel API pour mettre à jour le profil
    await authStore.updateProfile(form.value)
    
    toast.success('Profil mis à jour avec succès')
    emit('profile-updated')
    closeModal()
  } catch (error) {
    console.error('Erreur lors de la mise à jour du profil:', error)
    toast.error('Erreur lors de la mise à jour du profil')
  } finally {
    loading.value = false
  }
}

// Lifecycle
onMounted(() => {
  initializeForm()
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
  max-width: 600px;
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

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 20px;
}

.form-group label {
  color: #2c3e50;
  font-weight: 500;
  font-size: 0.9rem;
}

.form-group input,
.form-group select,
.form-group textarea {
  padding: 12px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.3s ease;
  font-family: inherit;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
}

.form-group textarea {
  resize: vertical;
  min-height: 100px;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 16px;
  margin-top: 32px;
  padding-top: 20px;
  border-top: 1px solid #e1e5e9;
}

.btn-cancel,
.btn-save {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
}

.btn-cancel {
  background: #f8f9fa;
  color: #6c757d;
  border: 2px solid #e1e5e9;
}

.btn-cancel:hover {
  background: #e9ecef;
  border-color: #adb5bd;
}

.btn-save {
  background: #667eea;
  color: white;
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-save:hover:not(:disabled) {
  background: #5a6fd8;
}

.btn-save:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading-spinner-small {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .modal-overlay {
    padding: 10px;
  }
  
  .modal-content {
    max-height: 95vh;
  }
  
  .form-row {
    grid-template-columns: 1fr;
    gap: 0;
  }
  
  .form-actions {
    flex-direction: column;
  }
  
  .btn-cancel,
  .btn-save {
    width: 100%;
  }
}
</style>
