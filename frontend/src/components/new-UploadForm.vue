<script setup>
import Dropzone from 'dropzone'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnnotationStore } from '@/stores/annotationStore'

const annotationStore = useAnnotationStore()
const uploadForm = ref(null)
const manuscriptName = ref('')
const models = ref([])
const modelSelected = ref('')

const message = ref('')
const messageType = ref('')
const isLoading = ref(false)

const router = useRouter()

// Get user info from localStorage
const user = JSON.parse(localStorage.getItem('user') || '{}')
const userId = user.userid || ''
const username = user.username || ''

const showMessage = (text, type = 'error') => {
  message.value = text
  messageType.value = type
  setTimeout(() => {
    message.value = ''
    messageType.value = ''
  }, 3000)
}

// Fetch models from backend
fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((res) => res.json())
  .then((data) => {
    models.value = data
  })

// Function to create manuscript via API
// Function to create manuscript via API
const createManuscript = async (fileNames = []) => {
  if (!manuscriptName.value || !modelSelected.value) {
    showMessage('Please fill manuscript name and select model.')
    return false
  }

  const payload = {
    userid: userId,
    username: username,
    manuscript_name: manuscriptName.value,
    model_selected: modelSelected.value,
    fileimagename: fileNames, // <<--- added here
    created_at: new Date().toISOString()
  }

  try {
    const res = await fetch(import.meta.env.VITE_BACKEND_URL + '/manuscripts/add-manuscript', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })

    if (!res.ok) {
      const err = await res.json()
      showMessage(err.message || 'Failed to create manuscript')
      return false
    }

    return true
  } catch (err) {
    showMessage('Network error: ' + err.message)
    return false
  }
}

// Handle submit button click
const handleSubmit = async () => {
  isLoading.value = true
  // Don’t create manuscript here yet — wait until files are uploaded
  uploadForm.value.processQueue() 
}

onMounted(() => {
  uploadForm.value = new Dropzone('#upload-form', {
    url: import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript',
    uploadMultiple: true,
    autoProcessQueue: false,
    parallelUploads: Infinity,
  })

  uploadForm.value.on('sending', (file, xhr, formData) => {
    formData.append('manuscript_name', manuscriptName.value)
    formData.append('model', modelSelected.value)
  })

  uploadForm.value.on('successmultiple', async (files) => {
    const fileNames = files.map((file) => file.name)

    // Now create manuscript with fileimagename
    const success = await createManuscript(fileNames)
    isLoading.value = false

    if (!success) return

    annotationStore.reset()
    annotationStore.modelName = modelSelected.value
    annotationStore.recognitions[manuscriptName.value] = {}

    const uploadedPageIds = files
      .filter((file) => file.status === Dropzone.SUCCESS && file.name)
      .map((file) => file.name.split('.')[0])
      .filter((id) => id.trim() !== '')

    uploadedPageIds.forEach((pageId) => {
      annotationStore.recognitions[manuscriptName.value][pageId] = {}
    })

    annotationStore.userAnnotations.push({
      manuscript_name: manuscriptName.value,
      selected_model: modelSelected.value,
      annotations: {},
    })

    annotationStore.setInitialPage()
    router.push({ name: 'new-semi-segment' })
  })

  uploadForm.value.on('error', () => {
    isLoading.value = false
  })
})

</script>

<template>
  <div class="upload-container">
    <Button class="back-btn" @click="router.go(-1)"> ← Back <span >(dashboard)</span></Button>
    <div class="upload-containerinner">
      <h2 class="title">Upload Manuscript</h2>

      <div v-if="message" :class="['message-box', messageType]">
        {{ message }}
      </div>

      <div class="form-dropzone-row">
        <div class="form-section">
          <div class="form-group">
            <label for="manuscriptName">Manuscript Name</label>
            <input
              type="text"
              id="manuscriptName"
              v-model="manuscriptName"
              placeholder="Enter manuscript name"
              class="input-field"
            />
          </div>

          <div class="form-group">
            <label for="model">Select Model</label>
            <select id="model" v-model="modelSelected" class="select-field">
              <option disabled value="">Select a model</option>
              <option v-for="model in models" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
          </div>
        </div>

        <div class="dropzone-section">
          <form id="upload-form" class="dropzone-box">
            <div class="dz-message">
              <p>Drag & Drop files here or click to upload</p>
            </div>
          </form>
        </div>
      </div>

      <div class="btn-row">
        <button
          class="action-btn primary-btn"
          @click="handleSubmit"
          :disabled="!manuscriptName || !modelSelected || isLoading"
        >
          <span v-if="!isLoading">Submit</span>
          <span v-else>Uploading...</span>
        </button>
      </div>
    </div>

    <!-- Loading overlay -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
      <p>Uploading files, please wait...</p>
    </div>
  </div>
</template>
<style scoped>
.back-btn {
  border-radius: 4px;
  background-color: #3b82f6;
  border: none;
  color: #ffffff;
  padding: 8px 20px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 18px;
  margin: 4px 2px;
  cursor: pointer;
  
}
.upload-container {
  width:60%;
  margin:  auto;
  background-color: #ffffff;
  border-radius: 16px;
 
}
.upload-containerinner {
  width: 100%;
  margin: 3rem auto;
  padding: 2.5rem;
  background-color: #ffffff;
  border-radius: 16px;
  
  box-shadow: 0 0px 08px rgba(0, 0, 0, 0.1);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.title {
  font-size: 2.2rem;
  font-weight: bold;
  margin-bottom: 2.5rem;
  color: #333;
  text-align: center;
}

.message-box {
  padding: 1rem;
  margin-bottom: 1.5rem;
  border-radius: 8px;
  text-align: center;
  font-weight: bold;
}

.message-box.error {
  background-color: #ffebee;
  color: #c62828;
}

.form-dropzone-row {
  display: flex;
  gap: 2rem;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
}

.form-section {
  flex: 1;
}

.dropzone-section {
  flex: 1;
}

.form-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 2rem;
}

.form-group label {
  margin-bottom: 0.75rem;
  font-weight: 600;
  font-size: 1.1rem;
  color: #555;
}

.input-field,
.select-field {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
  color: black;
  background-color: #fafafa;
  transition: all 0.3s ease;
}

.input-field:focus,
.select-field:focus {
  border-color: #4caf50;
  box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.1);
  background-color: #fff;
}

.dropzone-box {
  border: 2px dashed #4caf50;
  background-color: #f7fff7;
  border-radius: 12px;
  padding: 3.5rem 2rem;
  cursor: pointer;
  text-align: center;
  transition: all 0.3s ease;
}

.dropzone-box:hover {
  background-color: #e8f5e9;
}

.dz-message {
  font-size: 1.1rem;
  color: #4caf50;
}

.dz-message p {
  margin: 0;
}

.btn-row {
  display: flex;
  justify-content: flex-end;
}

.action-btn {
  cursor: pointer;
  padding: 10px 24px;
  font-size: 1.1em;
  font-weight: bold;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  border: 2px solid transparent;
}

.primary-btn {
  color: white;
  background-color: #4caf50;
  border-color: #4caf50;
}

.primary-btn:hover:not(:disabled) {
  background-color: #45a049;
  border-color: #45a049;
  transform: translateY(-2px);
  box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}

.primary-btn:disabled {
  background-color: #a5d6a7;
  border-color: #a5d6a7;
  cursor: not-allowed;
  box-shadow: none;
}

/* Loading overlay */
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

/* Responsive adjustments */
@media (max-width: 768px) {
  .form-dropzone-row {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
