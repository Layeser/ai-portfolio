<template>
  <div class="modal-overlay" @click="closeModal">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>Statistiques de la plateforme</h3>
        <button @click="closeModal" class="close-btn">
          <i class="fas fa-times"></i>
        </button>
      </div>
      
      <div class="modal-body">
        <!-- Filtres de période -->
        <div class="period-filters">
          <div class="filter-row">
            <div class="filter-group">
              <label>Période</label>
              <select v-model="selectedPeriod" @change="updateStatistics">
                <option value="7">7 derniers jours</option>
                <option value="30">30 derniers jours</option>
                <option value="90">3 derniers mois</option>
                <option value="365">1 an</option>
                <option value="all">Tout</option>
              </select>
            </div>
            
            <div class="filter-group">
              <label>Type de données</label>
              <select v-model="selectedDataType" @change="updateStatistics">
                <option value="users">Utilisateurs</option>
                <option value="activities">Activités</option>
                <option value="geographic">Géographique</option>
                <option value="academic">Académique</option>
              </select>
            </div>
            
            <div class="filter-group">
              <label>Format d'export</label>
              <select v-model="exportFormat">
                <option value="pdf">PDF</option>
                <option value="excel">Excel</option>
                <option value="csv">CSV</option>
              </select>
            </div>
          </div>
        </div>
        
        <!-- Statistiques générales -->
        <div class="stats-overview">
          <h4>Vue d'ensemble</h4>
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-number">{{ stats.totalUsers }}</div>
              <div class="stat-label">Utilisateurs totaux</div>
              <div class="stat-change" :class="getChangeClass(stats.userGrowth)">
                {{ stats.userGrowth > 0 ? '+' : '' }}{{ stats.userGrowth }}%
              </div>
            </div>
            
            <div class="stat-item">
              <div class="stat-number">{{ stats.activeUsers }}</div>
              <div class="stat-label">Utilisateurs actifs</div>
              <div class="stat-change" :class="getChangeClass(stats.activeGrowth)">
                {{ stats.activeGrowth > 0 ? '+' : '' }}{{ stats.activeGrowth }}%
              </div>
            </div>
            
            <div class="stat-item">
              <div class="stat-number">{{ stats.newUsers }}</div>
              <div class="stat-label">Nouveaux cette période</div>
              <div class="stat-change positive">+{{ stats.newUserGrowth }}%</div>
            </div>
            
            <div class="stat-item">
              <div class="stat-number">{{ stats.approvalRate }}%</div>
              <div class="stat-label">Taux d'approbation</div>
              <div class="stat-change" :class="getChangeClass(stats.approvalChange)">
                {{ stats.approvalChange > 0 ? '+' : '' }}{{ stats.approvalChange }}%
              </div>
            </div>
          </div>
        </div>
        
        <!-- Graphiques -->
        <div class="charts-section">
          <h4>Évolution dans le temps</h4>
          <div class="charts-grid">
            <div class="chart-container">
              <h5>Inscriptions par jour</h5>
              <div id="registrationsChart" class="chart"></div>
            </div>
            
            <div class="chart-container">
              <h5>Connexions par jour</h5>
              <div id="loginsChart" class="chart"></div>
            </div>
          </div>
        </div>
        
        <!-- Répartition géographique -->
        <div class="geographic-section" v-if="selectedDataType === 'geographic'">
          <h4>Répartition géographique</h4>
          <div class="geographic-chart">
            <div id="mapChart" class="chart"></div>
          </div>
        </div>
        
        <!-- Répartition académique -->
        <div class="academic-section" v-if="selectedDataType === 'academic'">
          <h4>Répartition académique</h4>
          <div class="academic-charts">
            <div class="chart-container">
              <h5>Filières CPGE</h5>
              <div id="filiereChart" class="chart"></div>
            </div>
            
            <div class="chart-container">
              <h5>Promotions</h5>
              <div id="promotionChart" class="chart"></div>
            </div>
          </div>
        </div>
        
        <!-- Actions -->
        <div class="actions-section">
          <button @click="exportStatistics" class="btn-export">
            <i class="fas fa-download me-2"></i>
            Exporter les statistiques ({{ exportFormat.toUpperCase() }})
          </button>
          
          <button @click="refreshStatistics" class="btn-refresh">
            <i class="fas fa-sync-alt me-2"></i>
            Actualiser
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useToast } from 'vue-toastification'

const emit = defineEmits(['close'])

const toast = useToast()

// État local
const selectedPeriod = ref('30')
const selectedDataType = ref('users')
const exportFormat = ref('pdf')

const stats = ref({
  totalUsers: 0,
  activeUsers: 0,
  newUsers: 0,
  approvalRate: 0,
  userGrowth: 0,
  activeGrowth: 0,
  newUserGrowth: 0,
  approvalChange: 0
})

// Méthodes
const closeModal = () => {
  emit('close')
}

const getChangeClass = (change) => {
  if (change > 0) return 'positive'
  if (change < 0) return 'negative'
  return 'neutral'
}

const updateStatistics = async () => {
  try {
    // Appel API pour récupérer les statistiques selon les filtres
    await loadStatistics()
    await updateCharts()
  } catch (error) {
    toast.error('Erreur lors du chargement des statistiques')
  }
}

const loadStatistics = async () => {
  // Simulation des données - à remplacer par des appels API
  stats.value = {
    totalUsers: 1250,
    activeUsers: 890,
    newUsers: 45,
    approvalRate: 87,
    userGrowth: 12,
    activeGrowth: 8,
    newUserGrowth: 15,
    approvalChange: 3
  }
}

const updateCharts = async () => {
  // Mettre à jour les graphiques avec Plotly
  // Cette méthode sera implémentée pour recharger les graphiques
}

const exportStatistics = async () => {
  try {
    // Appel API pour exporter les statistiques
    toast.success(`Export des statistiques en ${exportFormat.value.toUpperCase()} lancé`)
  } catch (error) {
    toast.error('Erreur lors de l\'export')
  }
}

const refreshStatistics = async () => {
  try {
    await updateStatistics()
    toast.success('Statistiques actualisées')
  } catch (error) {
    toast.error('Erreur lors de l\'actualisation')
  }
}

// Watchers
watch([selectedPeriod, selectedDataType], () => {
  updateStatistics()
})

// Lifecycle
onMounted(() => {
  updateStatistics()
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
  max-width: 1000px;
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

.period-filters {
  margin-bottom: 32px;
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

.filter-group select {
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

.stats-overview {
  margin-bottom: 32px;
}

.stats-overview h4 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.2rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.stat-item {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
  border: 2px solid transparent;
  transition: all 0.3s ease;
}

.stat-item:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  color: #2c3e50;
  margin-bottom: 8px;
}

.stat-label {
  color: #7f8c8d;
  font-size: 0.9rem;
  margin-bottom: 12px;
}

.stat-change {
  font-size: 0.9rem;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 20px;
  display: inline-block;
}

.stat-change.positive {
  background: #d4edda;
  color: #155724;
}

.stat-change.negative {
  background: #f8d7da;
  color: #721c24;
}

.stat-change.neutral {
  background: #e2e3e5;
  color: #383d41;
}

.charts-section {
  margin-bottom: 32px;
}

.charts-section h4 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.2rem;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}

.chart-container {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.chart-container h5 {
  color: #2c3e50;
  margin: 0 0 16px 0;
  font-size: 1.1rem;
  text-align: center;
}

.chart {
  height: 300px;
  width: 100%;
}

.geographic-section,
.academic-section {
  margin-bottom: 32px;
}

.geographic-section h4,
.academic-section h4 {
  color: #2c3e50;
  margin-bottom: 20px;
  font-size: 1.2rem;
}

.geographic-chart {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.academic-charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}

.actions-section {
  display: flex;
  justify-content: center;
  gap: 16px;
  padding-top: 20px;
  border-top: 1px solid #e1e5e9;
}

.btn-export,
.btn-refresh {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-export {
  background: #28a745;
  color: white;
}

.btn-export:hover {
  background: #218838;
}

.btn-refresh {
  background: #17a2b8;
  color: white;
}

.btn-refresh:hover {
  background: #138496;
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
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .academic-charts {
    grid-template-columns: 1fr;
  }
  
  .actions-section {
    flex-direction: column;
  }
  
  .btn-export,
  .btn-refresh {
    width: 100%;
    justify-content: center;
  }
}
</style>

