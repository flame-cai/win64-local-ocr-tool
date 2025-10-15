<script setup>
import Dropzone from 'dropzone'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnnotationStore } from '@/stores/annotationStore'
import axios from 'axios'

const annotationStore = useAnnotationStore()
const uploadForm = ref(null)
const manuscriptName = ref('')
const models = ref([])
const modelSelected = ref('')
const skipLayout = ref(false)
const manuscriList = ref([])
const message = ref('')
const messageType = ref('')
const isLoading = ref(false)
const fileCount = ref(0)
const shownameerror = ref(false)

const router = useRouter()

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
const updateFileCountDisplay = (newCount) => {
  fileCount.value = newCount
  const badgeEl = document.querySelector('#custom-preview .dz-count-badge')
  if (badgeEl) {
    // Badge shows the count of *other* files (+N means N+1 files total)
    badgeEl.textContent = newCount > 1 ? `+${newCount - 1}` : ''
  }
}

fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((res) => res.json())
  .then((data) => {
    models.value = data
  })

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
    fileimagename: fileNames,
    created_at: new Date().toISOString(),
  }

  try {
    const res = await fetch(import.meta.env.VITE_BACKEND_URL + '/manuscripts/add-manuscript', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
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
const HandleuniqiManuscript = async () => {
  shownameerror.value = false

  if (!manuscriptName.value) {
    showMessage('Please enter a manuscript name.')
    return
  }

  try {
    const response = await fetch(import.meta.env.VITE_BACKEND_URL + '/uploaded-manuscripts')
    const manuscripts = await response.json()

    const exists = manuscripts.some(
      (manuscript) => manuscript.trim().toLowerCase() === manuscriptName.value.trim().toLowerCase(),
    )

    if (exists) {
      shownameerror.value = true
      showMessage('Manuscript name already exists.')
      return
    }

    handleSubmit()
  } catch (error) {
    console.error('Error fetching manuscripts:', error)
    showMessage('Failed to validate manuscript name.')
  }
}

const handleSubmit = () => {
  isLoading.value = true
  uploadForm.value.processQueue()
}

onMounted(() => {
  uploadForm.value = new Dropzone('#upload-form', {
    url: () =>
      skipLayout.value
        ? import.meta.env.VITE_BACKEND_URL + '/upload-manuscript'
        : import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript',
    uploadMultiple: true,
    autoProcessQueue: false,
    parallelUploads: Infinity,
    previewsContainer: '#custom-preview',
    clickable: true,
    thumbnailWidth: 120,
    thumbnailHeight: 120,
    previewTemplate: `
      <div class="dz-preview dz-file-preview">
        <img data-dz-thumbnail class="dz-thumb" />
        <div class="dz-count-badge"></div>
      </div>
    `,
  })

  // --- Custom preview logic ---
  let fileCount = 0

  uploadForm.value.on('addedfile', (file) => {
    fileCount++
    const currentFiles = uploadForm.value.getAcceptedFiles().length
    updateFileCountDisplay(currentFiles)

    const previewEl = document.querySelector('#custom-preview')
    const imgEl = previewEl.querySelector('.dz-thumb')
    const badgeEl = previewEl.querySelector('.dz-count-badge')

    // update thumbnail for the first file
    if (fileCount === 1) {
      uploadForm.value.emit('thumbnail', file, file.dataURL)
    }

    // update badge count
    if (badgeEl) badgeEl.textContent = fileCount > 1 ? `+${fileCount - 1}` : ''
  })

  uploadForm.value.on('removedfile', () => {
    fileCount--
    const currentFiles = uploadForm.value.getAcceptedFiles().length
    updateFileCountDisplay(currentFiles)
    const badgeEl = document.querySelector('.dz-count-badge')
    if (badgeEl) badgeEl.textContent = fileCount > 1 ? `+${fileCount - 1}` : ''
  })

  uploadForm.value.on('sending', (file, xhr, formData) => {
    formData.append('manuscript_name', manuscriptName.value)
    formData.append('model', modelSelected.value)
  })

  uploadForm.value.on('successmultiple', async (files) => {
    if (skipLayout.value) {
      // ---- OLD FLOW ----
      const fileNames = files.map((file) => file.name)
      const success = await createManuscript(fileNames)
      if (!success) {
        isLoading.value = false
        return
      }

      const response = JSON.parse(files[0].xhr.response)
      const manuscript_name = Object.values(response)[0][0].manuscript_name
      const selected_model = Object.values(response)[0][0].selected_model
      annotationStore.recognitions[manuscript_name] = {}

      for (const page of Object.keys(response)) {
        annotationStore.recognitions[manuscript_name][page] = {}
        for (const line of Object.keys(response[page])) {
          const lineData = response[page][line]
          const line_name = lineData.line
          annotationStore.recognitions[manuscript_name][page][line_name] = {
            predicted_label: lineData.predicted_label,
            image_path: lineData.image_path,
            confidence_score: lineData.confidence_score,
          }
        }
      }

      annotationStore.userAnnotations.push({
        manuscript_name,
        selected_model,
        annotations: {},
      })

      isLoading.value = false
      router.push({ name: 'annotation-section' })
    } else {
      // ---- NEW FLOW ----
      const fileNames = files.map((file) => file.name)
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
    }
  })

  uploadForm.value.on('error', () => {
    isLoading.value = false
  })
})
</script>

<template>
  <div class="upload-container">
    <Button class="back-btn" @click="router.go(-1)"> ← Back <span>(dashboard)</span></Button>
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
            <div class="total-count-display" v-if="fileCount > 0">
              Total Pages : <strong>{{ fileCount + 1 }}</strong>
            </div>
            <div id="custom-preview" class="custom-preview"></div>
          </form>
        </div>
      </div>

      <div class="form-group">
        <label class="checkbox-label">
          <input type="checkbox" v-model="skipLayout" />
          Automatic Layout Analysis
        </label>
      </div>

      <div class="btn-row">
        <button
          class="action-btn primary-btn"
          @click="HandleuniqiManuscript"
          :disabled="!manuscriptName || !modelSelected || isLoading"
        >
          <span v-if="!isLoading">Submit</span>
          <span v-else>Uploading...</span>
        </button>
      </div>
    </div>

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
  width: 60%;
  margin: auto;
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
.checkbox-label input[type='checkbox'] {
  width: 20px;
  height: 20px;
  margin-right: 8px;
  vertical-align: middle;
  cursor: pointer;
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
  justify-content: center;
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

/* FIX: Ensure only the first preview is visible and it is clipped */
.custom-preview {
  /* Set a fixed size for the preview area */
  width: 120px;
  height: 120px;
  margin: 1rem auto; /* Center it */
  position: relative;

  /* CRITICAL: Hide all content that overflows this fixed area. 
     This prevents the horizontally stitched image effect. */
  overflow: hidden;

  /* Use flexbox to center the content within the fixed box */
  display: flex;
  justify-content: center;
  align-items: center;

  /* Add a placeholder border to visualize the single slot */
  border: 2px solid #e0e0e0;
  border-radius: 8px;
}

/* CRITICAL: Hide all but the first child preview element */
#custom-preview .dz-preview {
  display: none !important;
  /* Dropzone adds a .dz-preview for every file. We hide all of them. */
}

#custom-preview .dz-preview:first-child {
  display: block !important;
  /* We explicitly show only the first .dz-preview element */
  width: 120px;
  height: 120px;
  /* Reset margins/paddings if any were added by Dropzone defaults */
  margin: 0;
  padding: 0;
  position: absolute; /* Allows for centering */
  top: 0;
  left: 0;
}
/* END FIX */

.dz-thumb {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  border: 2px solid #4caf50;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
.dz-count-badge {
  position: absolute;
  /* Position relative to the dz-preview element */
  bottom: 4px;
  right: 4px;
  background: #4caf50;
  color: white;
  font-size: 0.9rem;
  font-weight: bold;
  border-radius: 50%;
  width: 28px;
  height: 28px;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10; /* Ensure badge is on top of the image */
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
@media (max-width: 768px) {
  .form-dropzone-row {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
