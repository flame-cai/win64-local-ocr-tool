<script setup>
import Dropzone from 'dropzone'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnnotationStore } from '@/stores/annotationStore'

const annotationStore = useAnnotationStore()
const uploadForm = ref()
const manuscriptName = ref('')
const models = ref([])
const modelSelected = ref('')

const router = useRouter()

fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((res) => res.json())
  .then((data) => {
    models.value = data
  })

onMounted(() => {
  uploadForm.value = new Dropzone('#upload-form', {
    url: import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript',
    uploadMultiple: true,
    autoProcessQueue: false,
    parallelUploads: Infinity,
  })

  uploadForm.value.on('completemultiple', (files) => {
    const currentManuscriptName = manuscriptName.value
    const currentModel = modelSelected.value

    if (!currentManuscriptName) {
      alert('Please enter a manuscript name.')
      return
    }
    if (!currentModel) {
      alert('Please select a model.')
      return
    }
    if (files.length === 0) {
      alert('Please add files to upload.')
      return
    }

    annotationStore.reset()
    annotationStore.modelName = currentModel
    annotationStore.recognitions[currentManuscriptName] = {}

    const uploadedPageIds = files
      .filter((file) => file.status === Dropzone.SUCCESS && file.name)
      .map((file) => file.name.split('.')[0])
      .filter((id) => id.trim() !== '')

    uploadedPageIds.forEach((pageId) => {
      annotationStore.recognitions[currentManuscriptName][pageId] = {}
    })

    annotationStore.userAnnotations.push({
      manuscript_name: currentManuscriptName,
      selected_model: currentModel,
      annotations: {},
    })

    annotationStore.setInitialPage()
    router.push({ name: 'new-semi-segment' })
  })
})
</script>

<template>
  <div class="upload-container">
    <h2 class="title">Upload Manuscript</h2>

    <div class="form-group">
      <label for="manuscriptName">Manuscript Name</label>
      <input
        type="text"
        id="manuscriptName"
        v-model="manuscriptName"
        placeholder="Enter manuscript name"
      />
    </div>

    <div class="form-group">
      <label for="model">Select Model</label>
      <select id="model" v-model="modelSelected">
        <option disabled value="">Select a model</option>
        <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
      </select>
    </div>

    <form id="upload-form" class="dropzone">
      <div class="dz-message">
        <span>Drag & Drop files here or click to upload</span>
      </div>
    </form>

    <button
      class="btn-submit"
      @click="uploadForm.processQueue()"
      :disabled="!manuscriptName || !modelSelected"
    >
      Submit
    </button>
  </div>
  
</template>

<style scoped>
.upload-container {
  max-width: 700px;
  margin: 2rem auto;
  padding: 2rem;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.title {
  font-size: 1.8rem;
  font-weight: bold;
  margin-bottom: 1.5rem;
  color: #1e3a8a;
}

.form-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 1.5rem;
  text-align: left;
}

.form-group label {
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #333;
}

.form-group input,
.form-group select {
  padding: 0.6rem 1rem;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
  transition: border 0.3s, box-shadow 0.3s;
}

.form-group input:focus,
.form-group select:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.dropzone {
  border: 2px dashed #3b82f6;
  background-color: #f0f4ff;
  border-radius: 12px;
  padding: 2rem;
  margin-bottom: 1.5rem;
  cursor: pointer;
  transition: background-color 0.3s, border-color 0.3s;
}

.dropzone:hover {
  background-color: #e0ebff;
  border-color: #1e3a8a;
}

.dz-message {
  font-size: 1.1rem;
  color: #1e3a8a;
}

.btn-submit {
  background-color: #1e3a8a;
  color: white;
  padding: 0.7rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.3s, transform 0.2s;
}

.btn-submit:disabled {
  background-color: #a5b4fc;
  cursor: not-allowed;
}

.btn-submit:hover:not(:disabled) {
  background-color: #3b82f6;
  transform: translateY(-2px);
}
</style>
