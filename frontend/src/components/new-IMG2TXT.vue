<script setup>
import { useRouter } from 'vue-router'
import AnnotationPage from '@/components/AnnotationPage.vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import CharacterPalette from './characterPalette.vue'

const router = useRouter()
const annotationStore = useAnnotationStore();

const manuscript_name = Object.keys(annotationStore.recognitions)[0];
const manuscriptPages = Object.keys(annotationStore.recognitions[manuscript_name] || {}); // Ensure manuscript_name exists

if (manuscript_name && annotationStore.currentPage && manuscriptPages.includes(annotationStore.currentPage)) {
  // If currentPage in store is valid for the current manuscript, use it.
  // No change needed to annotationStore.currentPage here, it's already correct.
} else if (manuscript_name && manuscriptPages.length > 0) {
  // Otherwise, default to the first page of the current manuscript
  annotationStore.currentPage = manuscriptPages[0];
} else {
  // Handle case where there are no pages or no manuscript
  annotationStore.currentPage = null;
  console.warn('No pages found for manuscript or manuscript_name is missing in new-IMG2TXT.vue');
}

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
  router.push({ name: 'new-semi-segment' }) // TODO we want to go directly here, with the manuscript name, and current page
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
    <button class="btn btn-warning me-2" @click="switchToSemiAutoSegmentation">Semi Segmentation</button>
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
