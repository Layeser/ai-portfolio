<template>
  <div class="supervisor-dashboard">
    <div class="dashboard-header">
      <h1><i class="fas fa-chart-line me-3"></i>Tableau de Bord CPGE Guinée</h1>
      <p>Visualisation des données des anciens étudiants des classes préparatoires</p>
    </div>

    <!-- Statistiques clés -->
    <div class="stats-overview">
      <div class="stat-card">
        <h5>Total Boursiers</h5>
        <p>{{ stats.totalBoursiers }}</p>
      </div>
      <div class="stat-card">
        <h5>Diplômés</h5>
        <p>{{ stats.totalDiplomes }}</p>
      </div>
      <div class="stat-card">
        <h5>En cours d'études</h5>
        <p>{{ stats.totalEtudiants }}</p>
      </div>
      <div class="stat-card">
        <h5>Intention de retour</h5>
        <p>{{ stats.intentionRetour }}%</p>
      </div>
    </div>

    <!-- Barre de filtres -->
    <div class="filter-bar">
      <div class="filter-row">
        <div class="filter-group">
          <label for="promotion-filter">Promotion</label>
          <select id="promotion-filter" v-model="filters.promotion" @change="applyFilters">
            <option value="">Toutes les promotions</option>
            <option v-for="promo in promoYears" :key="promo" :value="promo">{{ promo }}</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="filiere-filter">Filière</label>
          <select id="filiere-filter" v-model="filters.filiere" @change="applyFilters">
            <option value="">Toutes les filières</option>
            <option v-for="filiere in filieres" :key="filiere" :value="filiere">{{ filiere }}</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="pays-filter">Pays actuel</label>
          <select id="pays-filter" v-model="filters.pays" @change="applyFilters">
            <option value="">Tous les pays</option>
            <option v-for="pays in paysList" :key="pays" :value="pays">{{ pays }}</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="status-filter">Statut</label>
          <select id="status-filter" v-model="filters.status" @change="applyFilters">
            <option value="">Tous les statuts</option>
            <option value="diplome">Diplômé</option>
            <option value="etudiant">Étudiant</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Graphiques -->
    <div class="charts-grid">
      <!-- Statut des boursiers -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-user-graduate me-2"></i>Statut des boursiers</h5>
        </div>
        <div class="chart-body">
          <div id="statusChart" class="chart-container"></div>
        </div>
      </div>
      
      <!-- Répartition géographique -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-globe me-2"></i>Répartition géographique</h5>
        </div>
        <div class="chart-body">
          <div id="countryChart" class="chart-container"></div>
        </div>
      </div>
      
      <!-- Promotions par année -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-calendar-alt me-2"></i>Promotions par année</h5>
        </div>
        <div class="chart-body">
          <div id="promoChart" class="chart-container"></div>
        </div>
      </div>
      
      <!-- Secteurs d'activité -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-briefcase me-2"></i>Secteurs d'activité</h5>
        </div>
        <div class="chart-body">
          <div id="sectorChart" class="chart-container"></div>
        </div>
      </div>
      
      <!-- Établissements d'études -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-university me-2"></i>Établissements d'études</h5>
        </div>
        <div class="chart-body">
          <div id="etabChart" class="chart-container"></div>
        </div>
      </div>
      
      <!-- Intention de retour -->
      <div class="chart-card">
        <div class="chart-header">
          <h5><i class="fas fa-arrow-return-left me-2"></i>Intention de retour</h5>
        </div>
        <div class="chart-body">
          <div id="retourChart" class="chart-container"></div>
        </div>
      </div>
    </div>

    <!-- Tableau des étudiants -->
    <div class="students-table-section">
      <div class="table-header">
        <h3>Liste des étudiants</h3>
        <div class="table-actions">
          <button @click="exportStudents" class="btn-export">
            <i class="fas fa-download me-2"></i>Exporter
          </button>
        </div>
      </div>
      
      <div class="table-container">
        <table class="students-table">
          <thead>
            <tr>
              <th>Nom</th>
              <th>Email</th>
              <th>Statut</th>
              <th>Pays actuel</th>
              <th>Promotion</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="student in filteredStudents" :key="student.id">
              <td>{{ student.first_name }} {{ student.last_name }}</td>
              <td>{{ student.email }}</td>
              <td>
                <span :class="getStatusClass(student.boursier_status)">
                  {{ getStatusLabel(student.boursier_status) }}
                </span>
              </td>
              <td>{{ student.pays_actuel || 'Non renseigné' }}</td>
              <td>{{ student.promotion_cpge || 'Non renseigné' }}</td>
              <td>
                <button @click="viewStudentDetails(student.id)" class="btn-view">
                  <i class="fas fa-eye"></i>
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useToast } from 'vue-toastification'
import Plotly from 'plotly.js-dist'

const toast = useToast()

// État local
const stats = ref({
  totalBoursiers: 0,
  totalDiplomes: 0,
  totalEtudiants: 0,
  intentionRetour: 0
})

const filters = ref({
  promotion: '',
  filiere: '',
  pays: '',
  status: ''
})

const promoYears = ref([])
const filieres = ref([])
const paysList = ref([])
const students = ref([])

// Computed
const filteredStudents = computed(() => {
  let filtered = students.value

  if (filters.value.promotion) {
    filtered = filtered.filter(s => s.promotion_cpge === filters.value.promotion)
  }
  
  if (filters.value.filiere) {
    filtered = filtered.filter(s => s.filiere_cpge === filters.value.filiere)
  }
  
  if (filters.value.pays) {
    filtered = filtered.filter(s => s.pays_actuel === filters.value.pays)
  }
  
  if (filters.value.status) {
    filtered = filtered.filter(s => s.boursier_status === filters.value.status)
  }
  
  return filtered
})

// Méthodes
const loadDashboardData = async () => {
  try {
    // Simulation des données - à remplacer par des appels API
    stats.value = {
      totalBoursiers: 150,
      totalDiplomes: 89,
      totalEtudiants: 61,
      intentionRetour: 67
    }
    
    promoYears.value = ['2020', '2021', '2022', '2023', '2024']
    filieres.value = ['MP', 'PC', 'PSI', 'TSI', 'BCPST']
    paysList.value = ['France', 'Canada', 'États-Unis', 'Allemagne', 'Suisse']
    
    students.value = [
      {
        id: 1,
        first_name: 'John',
        last_name: 'Doe',
        email: 'john@example.com',
        boursier_status: 'diplome',
        pays_actuel: 'France',
        promotion_cpge: '2022',
        filiere_cpge: 'MP'
      },
      // Plus d'étudiants...
    ]
    
    // Initialiser les graphiques après le chargement des données
    nextTick(() => {
      initializeCharts()
    })
  } catch (error) {
    toast.error('Erreur lors du chargement des données')
  }
}

const applyFilters = () => {
  // Les filtres sont appliqués automatiquement via computed
  // Recharger les graphiques avec les données filtrées
  updateCharts()
}

const initializeCharts = () => {
  // Graphique du statut des boursiers
  const statusData = [
    { values: [stats.value.totalDiplomes, stats.value.totalEtudiants], 
      labels: ['Diplômés', 'Étudiants'], 
      type: 'pie',
      marker: { colors: ['#28a745', '#17a2b8'] }
    }
  ]
  
  Plotly.newPlot('statusChart', statusData, {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    showlegend: true,
    legend: { orientation: 'h', x: 0.5, y: -0.1 }
  })
  
  // Graphique de répartition géographique
  const countryData = [
    { x: ['France', 'Canada', 'États-Unis', 'Allemagne'], 
      y: [45, 23, 18, 12], 
      type: 'bar',
      marker: { color: '#667eea' }
    }
  ]
  
  Plotly.newPlot('countryChart', countryData, {
    margin: { t: 20, b: 40, l: 40, r: 20 },
    xaxis: { title: 'Pays' },
    yaxis: { title: 'Nombre d\'étudiants' }
  })
  
  // Autres graphiques...
}

const updateCharts = () => {
  // Mettre à jour les graphiques avec les données filtrées
  // Cette méthode sera implémentée pour recharger les graphiques
}

const getStatusLabel = (status) => {
  const labels = {
    'diplome': 'Diplômé',
    'etudiant': 'Étudiant'
  }
  return labels[status] || status
}

const getStatusClass = (status) => {
  const classes = {
    'diplome': 'status-diplome',
    'etudiant': 'status-etudiant'
  }
  return classes[status] || 'status-default'
}

const viewStudentDetails = (studentId) => {
  // Navigation vers la page de détails de l'étudiant
  console.log('Voir détails étudiant:', studentId)
}

const exportStudents = async () => {
  try {
    // Appel API pour exporter les données des étudiants
    toast.success('Export des données lancé')
  } catch (error) {
    toast.error('Erreur lors de l\'export')
  }
}

// Lifecycle
onMounted(() => {
  loadDashboardData()
})

// Watchers
watch(filters, () => {
  updateCharts()
}, { deep: true })
</script>

<style scoped>
.supervisor-dashboard {
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

.stats-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  text-align: center;
}

.stat-card h5 {
  color: #7f8c8d;
  margin: 0 0 8px 0;
  font-size: 0.9rem;
  font-weight: 500;
}

.stat-card p {
  color: #2c3e50;
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
}

.filter-bar {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  margin-bottom: 32px;
}

.filter-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.filter-group label {
  display: block;
  margin-bottom: 8px;
  color: #2c3e50;
  font-weight: 500;
}

.filter-group select {
  width: 100%;
  padding: 10px 12px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.3s ease;
}

.filter-group select:focus {
  outline: none;
  border-color: #667eea;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.chart-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.chart-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 20px;
}

.chart-header h5 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.chart-body {
  padding: 20px;
}

.chart-container {
  height: 300px;
  width: 100%;
}

.students-table-section {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid #e1e5e9;
}

.table-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 1.3rem;
}

.btn-export {
  background: #28a745;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.btn-export:hover {
  background: #218838;
}

.table-container {
  overflow-x: auto;
}

.students-table {
  width: 100%;
  border-collapse: collapse;
}

.students-table th,
.students-table td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid #e1e5e9;
}

.students-table th {
  background: #f8f9fa;
  color: #2c3e50;
  font-weight: 600;
  font-size: 0.9rem;
}

.students-table td {
  color: #2c3e50;
}

.status-diplome {
  background: #d4edda;
  color: #155724;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-etudiant {
  background: #d1ecf1;
  color: #0c5460;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-default {
  background: #e2e3e5;
  color: #383d41;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.btn-view {
  background: #17a2b8;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.btn-view:hover {
  background: #138496;
}

@media (max-width: 768px) {
  .supervisor-dashboard {
    padding: 16px;
  }
  
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .filter-row {
    grid-template-columns: 1fr;
  }
  
  .table-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }
}
</style>
