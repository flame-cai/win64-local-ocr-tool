<script setup>
import Dropzone from 'dropzone'

import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnnotationStore } from '@/stores/annotationStore'

const annotationStore = useAnnotationStore()
const uploadForm = ref()
const manuscriptName = ref()
const models = ref([])
const modelSelected = ref('')

const router = useRouter()

fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((response) => response.json())
  .then((object) => {
    models.value = object
  })

onMounted(() => {
  uploadForm.value = new Dropzone('#upload-form', {
    uploadMultiple: true,
    autoProcessQueue: false,
    parallelUploads: Infinity,
  })
  uploadForm.value.on('completemultiple', function (files) {
    const response = JSON.parse(files[0].xhr.response)
    const manuscript_name = Object.values(response)[0][0].manuscript_name
    const selected_model = Object.values(response)[0][0].selected_model
    annotationStore.recognitions[manuscript_name] = {}

    for (const page of Object.keys(response)) {
      annotationStore.recognitions[manuscript_name][page] = {}
      for (const line in response[page]) {
        const line_name = response[page][line]['line']
        annotationStore.recognitions[manuscript_name][page][line_name] = {}
        annotationStore.recognitions[manuscript_name][page][line_name]['predicted_label'] =
          response[page][line]['predicted_label']
        annotationStore.recognitions[manuscript_name][page][line_name]['image_path'] =
          response[page][line]['image_path']
        annotationStore.recognitions[manuscript_name][page][line_name]['confidence_score'] =
          response[page][line]['confidence_score']
      }
    }
    annotationStore.userAnnotations.push({
      manuscript_name: manuscript_name,
      selected_model: selected_model,
      annotations: {},
    })

    router.push({ name: 'annotation-section' })
  })
})

const UPLOAD_URL = import.meta.env.VITE_BACKEND_URL + '/upload-manuscript'

</script>

<template>
  <div class="mb-3">
    <label for="manuscriptName" class="form-label">Manuscript Name</label>
    <input type="text" class="form-control" id="manuscriptName" v-model="manuscriptName" />
  </div>
  <div class="mb-3">
    <label for="model" class="form-label">Model</label>
    <select class="form-select" id="model" v-model="modelSelected" placeholder="Select a model">
      <option disabled hidden value="">Select a model</option>
      <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
    </select>
  </div>
  <form :action="UPLOAD_URL" class="dropzone" id="upload-form">
    <div class="previews"></div>
    <input type="hidden" name="manuscript_name" :value="manuscriptName" />
    <input type="hidden" name="model" :value="modelSelected" />
  </form>
  <button @click="uploadForm.processQueue()" class="btn btn-primary mt-3">Submit</button>
</template>

<style>

form.dropzone {
  background-color: var(--bs-body-bg);
  border: var(--bs-border-width) solid var(--bs-border-color);
  color: var(--bs-body-color);
  font-family: inherit;
}

</style>