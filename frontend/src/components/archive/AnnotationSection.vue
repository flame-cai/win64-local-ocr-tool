<script setup>
import { useRouter } from 'vue-router'
import AnnotationPage from '@/components/archive/AnnotationPage.vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import CharacterPalette from '../typing-utils/characterPalette.vue'

const router = useRouter()
const annotationStore = useAnnotationStore()
const manuscript_name = Object.keys(annotationStore.recognitions)[0]
annotationStore.currentPage = Object.keys(annotationStore.recognitions[manuscript_name])[0]

function uploadGroundTruth() {
  // FIX 1: Run the calculation FIRST. This ensures the main userAnnotations object in the
  // store is populated with the levenshtein_distance for every edited line.
  annotationStore.calculateLevenshteinDistances()

  const originalRecognitions = annotationStore.recognitions[manuscript_name]
  const userEditedAnnotations = annotationStore.userAnnotations[0].annotations
  const annotationsToSend = {}

  for (const pageName in userEditedAnnotations) {
    for (const lineName in userEditedAnnotations[pageName]) {
      const userEditObject = userEditedAnnotations[pageName][lineName]
      const userEditText = userEditObject?.ground_truth
      const originalPrediction = originalRecognitions[pageName]?.[lineName]?.predicted_label

      // The filter condition remains the same: only include lines that were actually changed.
      if (userEditText && userEditText !== originalPrediction) {
        if (!annotationsToSend[pageName]) {
          annotationsToSend[pageName] = {}
        }
        // FIX 2: Copy the ENTIRE line object from the store, not just the ground_truth.
        // This object now includes the 'levenshtein_distance' calculated above.
        annotationsToSend[pageName][lineName] = userEditObject
      }
    }
  }

  // This safety check is still critical and correct.
  if (Object.keys(annotationsToSend).length === 0) {
    alert('No changes have been made to the annotations. Please edit a line before fine-tuning.')
    return
  }

  const payload = [
    {
      ...annotationStore.userAnnotations[0],
      annotations: annotationsToSend,
      model_name: annotationStore.modelName,
    },
  ]

  fetch(import.meta.env.VITE_BACKEND_URL + '/fine-tune', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
    .then((response) => {
      if (response.ok) {
        annotationStore.reset()
        window.location.href = '/dashboard'
      } else {
        alert('An error occurred during the fine-tuning process. Please check the backend logs.')
      }
    })
    .catch((error) => {
      console.error('Fetch error:', error)
      alert('Failed to connect to the backend.')
    })
}

// function switchToSemiAutoSegmentation() {
//   router.push({ name: 'semi-segment' })
// }
</script>
<template>
  <header class="header-section">
    <div class="d-flex align-items-center gap-3">
      <div class="input-group">
        <span class="input-group-text" id="model-name-label">Model name</span>
        <input
          type="text"
          class="form-control"
          placeholder="Name your model..."
          aria-label="Model name"
          aria-describedby="model-name-label"
          v-model="annotationStore.modelName"
        />
      </div>
    </div>

    <div class="d-flex align-items-center gap-2">
      <select class="form-select" id="page" v-model="annotationStore.currentPage">
        <option disabled value="">Select a page...</option>
        <option
          v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
          :key="page_name"
          :value="page_name"
        >
          {{ page_name }}
        </option>
      </select>
      <button class="btn btn-primary" @click="uploadGroundTruth">Fine-tune</button>
      <button class="btn btn-success" @click="annotationStore.exportToTxt">Export</button>
      <CharacterPalette />
    </div>
  </header>

  <AnnotationPage
    v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
    :key="page_data"
    :data="page_data"
    :page_name="page_name"
    :manuscript_name="manuscript_name"
    v-show="annotationStore.currentPage === page_name"
  />
</template>

<style scoped>
.header-section {
  width: 100%;
  display: flex;
  flex-wrap: nowrap;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 12px 24px;
  background-color: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  box-sizing: border-box;
}

.form-control,
.form-select {
  width: fit-content;
}
</style>
