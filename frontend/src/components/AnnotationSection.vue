<script setup>
import { useRouter } from 'vue-router'
import AnnotationPage from '@/components/AnnotationPage.vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import CharacterPalette from './characterPalette.vue'

const router = useRouter()
const annotationStore = useAnnotationStore()
const manuscript_name = Object.keys(annotationStore.recognitions)[0]
annotationStore.currentPage = Object.keys(annotationStore.recognitions[manuscript_name])[0]

function uploadGroundTruth() {
  annotationStore.calculateLevenshteinDistances()
  annotationStore.userAnnotations.forEach((elem) => {
    elem['model_name'] = annotationStore.modelName
    console.log('added Model name', annotationStore.modelName)
  })
  fetch(import.meta.env.VITE_BACKEND_URL + '/fine-tune', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(annotationStore.userAnnotations),
  }).then(() => {
    annotationStore.reset()
    router.push({ name: 'upload-manuscript' })
  })
}

function switchToSegmentation() {
  router.push({ name: 'segment' })
}

function switchToSemiAutoSegmentation() {
  router.push({ name: 'semi-segment' })
}

</script>

<template>
  <div class="mb-3">
    <label for="model-name" class="form-label">Model name</label>
    <input
      class="form-control"
      placeholder="Name your model..."
      v-model="annotationStore.modelName"
    />
  </div>
  <div class="mb-3">
    <button class="btn btn-primary me-2" @click="uploadGroundTruth">Fine-tune</button>
    <button class="btn btn-warning me-2" @click="switchToSegmentation">Correct Image Segments</button>
    <button class="btn btn-warning me-2" @click="switchToSemiAutoSegmentation">Test Segmentation</button>
    <button class="btn btn-success me-2" @click="annotationStore.exportToTxt">Export</button>
    <CharacterPalette />
  </div>
  <div class="mb-3">
    <label for="page" class="form-label">Page</label>
    <select
      class="form-select"
      id="page"
      v-model="annotationStore.currentPage"
      placeholder="Select a model"
    >
      <option
        v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
        :key="page_name"
        :value="page_name"
      >
        {{ page_name }}
      </option>
    </select>
  </div>
  <AnnotationPage
    v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
    :key="page_data"
    :data="page_data"
    :page_name="page_name"
    :manuscript_name="manuscript_name"
    v-show="annotationStore.currentPage === page_name"
  />
</template>
