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
  annotationStore.calculateLevenshteinDistances();

  const originalRecognitions = annotationStore.recognitions[manuscript_name];
  const userEditedAnnotations = annotationStore.userAnnotations[0].annotations;
  const annotationsToSend = {};
  
  for (const pageName in userEditedAnnotations) {
    for (const lineName in userEditedAnnotations[pageName]) {
      const userEditObject = userEditedAnnotations[pageName][lineName];
      const userEditText = userEditObject?.ground_truth;
      const originalPrediction = originalRecognitions[pageName]?.[lineName]?.predicted_label;
      
      // The filter condition remains the same: only include lines that were actually changed.
      if (userEditText && userEditText !== originalPrediction) {
        if (!annotationsToSend[pageName]) {
          annotationsToSend[pageName] = {};
        }
        // FIX 2: Copy the ENTIRE line object from the store, not just the ground_truth.
        // This object now includes the 'levenshtein_distance' calculated above.
        annotationsToSend[pageName][lineName] = userEditObject;
      }
    }
  }
  
  // This safety check is still critical and correct.
  if (Object.keys(annotationsToSend).length === 0) {
    alert("No changes have been made to the annotations. Please edit a line before fine-tuning.");
    return;
  }
  
  const payload = [{
    ...annotationStore.userAnnotations[0],
    annotations: annotationsToSend,
    model_name: annotationStore.modelName
  }];

  fetch(import.meta.env.VITE_BACKEND_URL + '/fine-tune', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  }).then((response) => {
    if (response.ok) {
        annotationStore.reset()
        router.push({ name: 'upload-manuscript' })
    } else {
        alert("An error occurred during the fine-tuning process. Please check the backend logs.");
    }
  }).catch(error => {
    console.error("Fetch error:", error);
    alert("Failed to connect to the backend.");
  })
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
    <h1 >  
      this is archive annotation page 
    </h1>
    <button class="btn btn-primary me-2" @click="uploadGroundTruth">Fine-tune</button>
    <button class="btn btn-warning me-2" @click="switchToSemiAutoSegmentation">Semi Automatic Segmentation</button>
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