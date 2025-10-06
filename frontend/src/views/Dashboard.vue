<!--  -->
<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { useAnnotationStore } from '@/stores/annotationStore'

const router = useRouter()
const annotationStore = useAnnotationStore()
const manuscripts = ref([])
const isLoading = ref(false) // Global loading overlay

const userid = JSON.parse(localStorage.getItem('user')).userid
const RECOGNITION_URL = import.meta.env.VITE_BACKEND_URL + '/recognise'
const fetchManuscriptsurl= import.meta.env.VITE_BACKEND_URL + '/manuscripts/get-manuscripts/' + userid

const goToNewManuscript = () => router.push('/new/upload')
const goToEditManuscript = (name, image) => router.push(`/edit/${name}/${image.split(".")[0]}`)

// Annotate Text button logic
const goToAnnotateManuscript = async (m) => {
  try {
    isLoading.value = true // Show loading overlay

    const response = await fetch(RECOGNITION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manuscript_name: m.manuscript_name, model: m.model_selected }),
    })

    const object = await response.json()
    const manuscript_name = Object.values(object)[0][0].manuscript_name
    const selected_model = Object.values(object)[0][0].selected_model

    // Seed recognitions
    annotationStore.recognitions[manuscript_name] = {}
    for (const page of Object.keys(object)) {
      annotationStore.recognitions[manuscript_name][page] = {}
      for (const line in object[page]) {
        const line_name = object[page][line]['line']
        annotationStore.recognitions[manuscript_name][page][line_name] = {
          predicted_label: object[page][line]['predicted_label'],
          image_path: object[page][line]['image_path'],
          confidence_score: object[page][line]['confidence_score'],
        }
      }
    }

    // Seed userAnnotations
    annotationStore.userAnnotations.push({
      manuscript_name,
      selected_model,
      annotations: {},
    })

    router.push({ name: 'annotation-section' })
  } catch (error) {
    console.error("Error fetching recognition:", error)
    alert("Failed to load manuscript for annotation.")
  } finally {
    isLoading.value = false // Hide loading overlay
  }
}

// Fetch manuscripts
const fetchManuscripts = async () => {
  try {
    const response = await axios.get(fetchManuscriptsurl)
    manuscripts.value = response.data.manuscripts || []
  } catch (error) {
    console.error("Error fetching manuscripts:", error)
  }
}

onMounted(() => {
  fetchManuscripts()
})
</script>

<template>
  <div class="main-container">
    <div class="header-section">
      <h1>Your Manuscripts</h1>
      <button class="action-btn primary-btn" @click="goToNewManuscript">+ New Manuscript</button>
    </div>

    <div class="manuscript-list-grid">
      <div v-if="manuscripts.length === 0" class="no-manuscripts">
        No manuscripts found.
      </div>

      <div v-for="m in manuscripts" :key="m.id" class="manuscript-card">
        <div class="manuscript-info">
          <p class="manuscript-name">{{ m.manuscript_name }}</p>
          <p class="manuscript-detail">Model: {{ m.model_selected }}</p>
          <p class="manuscript-detail">Image: {{ m.fileimagename}}</p>
          <p class="manuscript-detail">Date: {{ new Date(m.created_at).toLocaleDateString() }}</p>
        </div>
        <div class="manuscript-actions">
          <button class="action-btn" @click="goToEditManuscript(m.manuscript_name, m.fileimagename)">Edit Layout</button>
          <button class="action-btn secondary-btn" @click="goToAnnotateManuscript(m)">
            Annotate Text
          </button>
        </div>
      </div>
    </div>

    <!-- Loading overlay -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
      <p>Loading manuscript for annotation, please wait...</p>
    </div>
  </div>
</template>
<style scoped>
.main-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px;
  min-height: 100vh;
  width: 100%;
  background-color: #f7f9fc;
}

.header-section {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
  padding: 0 32px;
  box-sizing: border-box;
}

.header-section h1 {
  font-size: 1.8em;
  color: #333;
  text-transform: capitalize;
  margin: 0;
}

.action-btn {
  cursor: pointer;
  padding: 8px 16px;
  font-size: 1em;
  font-weight: 500;
  border-radius: 6px;
  transition: all 0.2s ease;
  margin-left: 10px;
  white-space: nowrap;
}

.primary-btn {
  color: white;
  background-color: #4caf50;
  border: 1px solid #4caf50;
}

.primary-btn:hover {
  background-color: #45a049;
  border-color: #45a049;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transform: translateY(-1px);
}

.secondary-btn {
  color: #4caf50;
  background-color: transparent;
  border: 1px solid #4caf50;
}

.secondary-btn:hover {
  background-color: #e8f5e9;
}

.manuscript-list-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  padding: 0 32px;
  box-sizing: border-box;
}

.manuscript-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 20px;
  background-color: white;
  border: 1px solid #ddd;
  border-top: 5px solid #4caf50;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition:
    transform 0.3s ease,
    box-shadow 0.3s ease;
}

.manuscript-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
  cursor: pointer;
}

.manuscript-info {
  flex-direction: column;
  gap: 8px;
  font-size: 0.9em;
  color: #555;
  margin-bottom: 20px;
}

.manuscript-name {
  font-weight: 700;
  color: #222;
  font-size: 1.2em;
  margin: 0 0 10px 0;
}

.manuscript-detail {
  margin: 0;
}

.manuscript-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.manuscript-card .action-btn {
  margin-left: 0;
}

@media (max-width: 1400px) {
  .manuscript-list-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 1024px) {
  .manuscript-list-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 600px) {
  .header-section {
    flex-direction: column;
    align-items: flex-start;
    padding: 0 16px;
  }

  .header-section h1 {
    margin-bottom: 10px;
  }

  .manuscript-list-grid {
    grid-template-columns: 1fr;
    gap: 15px;
    padding: 0 16px;
  }

  .manuscript-actions {
    justify-content: space-between;
  }
}
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.85);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
.spinner {
  border: 6px solid #f3f3f3;
  border-top: 6px solid #4caf50;
  border-radius: 50%;
  width: 48px;
  height: 48px;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
