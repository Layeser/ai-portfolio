<template>
  <div v-if="message" class="alert" :class="alertType">
    <div class="alert-content">
      <span class="alert-icon">{{ alertIcon }}</span>
      <span class="alert-message">{{ message }}</span>
      <button @click="$emit('close')" class="alert-close">
        <span>&times;</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  type: {
    type: String,
    default: 'info',
    validator: (value) => ['success', 'error', 'warning', 'info'].includes(value)
  },
  message: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['close'])

const alertType = computed(() => `alert-${props.type}`)

const alertIcon = computed(() => {
  const icons = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
  }
  return icons[props.type] || icons.info
})
</script>

<style scoped>
.alert {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  min-width: 300px;
  max-width: 500px;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  animation: slideIn 0.3s ease-out;
}

.alert-content {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  gap: 12px;
}

.alert-icon {
  font-size: 18px;
  flex-shrink: 0;
}

.alert-message {
  flex: 1;
  font-size: 14px;
  font-weight: 500;
}

.alert-close {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: inherit;
  opacity: 0.7;
  transition: opacity 0.2s ease;
  flex-shrink: 0;
}

.alert-close:hover {
  opacity: 1;
}

/* Alert types */
.alert-success {
  background: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}

.alert-error {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.alert-warning {
  background: #fff3cd;
  color: #856404;
  border: 1px solid #ffeaa7;
}

.alert-info {
  background: #d1ecf1;
  color: #0c5460;
  border: 1px solid #bee5eb;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .alert {
    top: 10px;
    right: 10px;
    left: 10px;
    min-width: auto;
  }
}
</style>
