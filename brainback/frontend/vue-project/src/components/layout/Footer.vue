<template>
  <footer class="footer">
    <div class="footer-container">
      <div class="footer-content">
        <div class="footer-section">
          <h3>BrainBack</h3>
          <p>Plateforme de mentorat et d'opportunités pour les étudiants et professionnels.</p>
        </div>
        
        <div class="footer-section">
          <h4>Liens rapides</h4>
          <ul class="footer-links">
            <li><router-link to="/dashboard">Tableau de bord</router-link></li>
            <li><router-link to="/opportunities">Opportunités</router-link></li>
            <li><router-link to="/mentorship">Mentorat</router-link></li>
          </ul>
        </div>
        
        <div class="footer-section">
          <h4>Support</h4>
          <ul class="footer-links">
            <li><a href="#help">Aide</a></li>
            <li><a href="#contact">Contact</a></li>
            <li><a href="#faq">FAQ</a></li>
          </ul>
        </div>
      </div>
      
      <div class="footer-bottom">
        <p>&copy; 2025 BrainBack. Tous droits réservés.</p>
      </div>
    </div>
  </footer>
</template>

<script setup>
// Footer component
</script>

<style scoped>
.footer {
  background: #2c3e50;
  color: white;
  padding: 40px 0 20px;
  margin-top: auto;
}

.footer-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.footer-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 40px;
  margin-bottom: 30px;
}

.footer-section h3 {
  color: #667eea;
  margin: 0 0 15px 0;
  font-size: 20px;
}

.footer-section h4 {
  color: #ecf0f1;
  margin: 0 0 15px 0;
  font-size: 16px;
}

.footer-section p {
  margin: 0;
  line-height: 1.6;
  color: #bdc3c7;
}

.footer-links {
  list-style: none;
  padding: 0;
  margin: 0;
}

.footer-links li {
  margin-bottom: 8px;
}

.footer-links a {
  color: #bdc3c7;
  text-decoration: none;
  transition: color 0.3s ease;
}

.footer-links a:hover {
  color: #667eea;
}

.footer-bottom {
  border-top: 1px solid #34495e;
  padding-top: 20px;
  text-align: center;
}

.footer-bottom p {
  margin: 0;
  color: #95a5a6;
  font-size: 14px;
}

@media (max-width: 768px) {
  .footer-content {
    grid-template-columns: 1fr;
    gap: 30px;
    text-align: center;
  }
}
</style>
