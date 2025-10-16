<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { useAnnotationStore } from '@/stores/annotationStore'
import { Trash2 } from 'lucide-vue-next'

const router = useRouter()
const annotationStore = useAnnotationStore()
const manuscripts = ref([])
const isLoading = ref(false)

const userid = JSON.parse(localStorage.getItem('user')).userid
const RECOGNITION_URL = import.meta.env.VITE_BACKEND_URL + '/recognise'
const fetchManuscriptsurl =
  import.meta.env.VITE_BACKEND_URL + '/manuscripts/get-manuscripts/' + userid
const deleteManuscripturl = import.meta.env.VITE_BACKEND_URL + '/manuscripts/delete-manuscript'
const check_manuscriptsaveURL =
  import.meta.env.VITE_BACKEND_URL + '/manuscripts/check-savemanuscript'

const goToNewManuscript = () => router.push('/new/upload')

const goToEditManuscript = (name, image, modelname) => {
  annotationStore.modelName = modelname
  const cleanString = image.replace(/^\[|\]$/g, '')
  const imageArray = cleanString.split(',').map((item) => item.trim())
  const firstImage = imageArray[0].split('.')[0].replace(/['"]+/g, '')
  router.push(`/edit/${name}/${firstImage}`)
}

const checkManuscriptLines = async (manuscript_name) => {
  try {
    const response = await axios.post(check_manuscriptsaveURL, {
      manuscript_name,
    })
    return response.data.exist
  } catch (error) {
    console.error('Error checking lines folder:', error)
    return false
  }
}

const fetchManuscripts = async () => {
  try {
    isLoading.value = true
    const response = await axios.get(fetchManuscriptsurl)
    manuscripts.value = response.data.manuscripts || []

    for (const m of manuscripts.value) {
      m.hasLines = await checkManuscriptLines(m.manuscript_name)
    }
  } catch (error) {
    console.error('Error fetching manuscripts:', error)
  } finally {
    isLoading.value = false
  }
}

const deleteManuscript = async (manuscript_name, model_selected) => {
  try {
    isLoading.value = true
    const deletereponse = await axios.delete(deleteManuscripturl, {
      data: { userid, manuscript_name, model_selected },
    })

    if (deletereponse.status === 200) {
      await fetchManuscripts()
    }
  } catch (error) {
    console.error('Error deleting manuscript:', error)
    alert('Failed to delete manuscript.')
  } finally {
    isLoading.value = false
  }
}

const goToAnnotateManuscript = async (m) => {
  try {
    isLoading.value = true
    const response = await fetch(RECOGNITION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manuscript_name: m.manuscript_name, model: m.model_selected }),
    })

    const object = await response.json()
    const manuscript_name = Object.values(object)[0][0].manuscript_name
    const selected_model = Object.values(object)[0][0].selected_model

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

    annotationStore.userAnnotations.push({
      manuscript_name,
      selected_model,
      annotations: {},
    })

    router.push({ name: 'annotation-section' })
  } catch (error) {
    console.error('Error fetching recognition:', error)
    alert('Failed to load manuscript for annotation.')
  } finally {
    isLoading.value = false
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
      <div v-if="manuscripts.length === 0" class="no-manuscripts">No manuscripts found.</div>

      <div v-for="m in manuscripts" :key="m.id" class="manuscript-card">
        <div class="manuscript-info">
          <div class="manuscript-name">
            <span class="manuscript-title">{{ m.manuscript_name }}</span>
            <button
              class="delete-btn"
              title="Delete"
              @click="deleteManuscript(m.manuscript_name, m.model_selected)"
            >
              <Trash2 class="delete-icon" />
            </button>
          </div>

          <p class="manuscript-detail">Model: {{ m.model_selected }}</p>
          <p class="manuscript-detail">
            Image:
            {{
              m.fileimagename
                ? m.fileimagename
                    .split('.')[0]
                    .replace(/['"\[\]]+/g, '')
                    .slice(0, 10) + (m.fileimagename.split('.')[0].length > 10 ? '...' : '')
                : ''
            }}
          </p>
          <p class="manuscript-detail">Date: {{ new Date(m.created_at).toLocaleDateString() }}</p>

          <p class="manuscript-detail">
            <span v-if="!m.hasLines" class="lines-missing">Layout unsaved ⚠️</span>
          </p>
        </div>

        <div class="manuscript-actions">
          <button
            class="action-btn"
            @click="goToEditManuscript(m.manuscript_name, m.fileimagename, m.model_selected)"
          >
            {{ !m.hasLines ? 'Save layout' : 'Edit layout' }}
          </button>
          <button
            class="action-btn secondary-btn"
            :disabled="!m.hasLines"
            :class="{ disabled: !m.hasLines }"
            @click="goToAnnotateManuscript(m)"
          >
            Annotate Text
          </button>
        </div>
      </div>
    </div>

    <!-- Loading overlay -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
      <p>Process is underway, please wait...</p>
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
}

.secondary-btn {
  color: #4caf50;
  background-color: transparent;
  border: 1px solid #4caf50;
}

.secondary-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
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
}

.manuscript-info {
  flex-direction: column;
  gap: 8px;
  font-size: 0.9em;
  color: #555;
  margin-bottom: 16px;
}

.manuscript-name {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.manuscript-title {
  font-weight: 700;
  font-size: 1.2em;
  color: #222;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.delete-btn {
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.delete-icon {
  width: 20px;
  height: 20px;
  color: #e53935;
  transition:
    color 0.2s ease,
    transform 0.2s ease;
}

.delete-btn:hover .delete-icon {
  color: #b71c1c;
  transform: scale(1.1);
}

.manuscript-detail {
  margin: 0;
}

.lines-ok {
  color: #2e7d32;
  font-weight: 600;
}

.lines-missing {
  color: #e53935;
  font-weight: 600;
}

.manuscript-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
  padding-top: 15px;
  border-top: 1px solid #eee;
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
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
