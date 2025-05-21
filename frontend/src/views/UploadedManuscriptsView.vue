<script setup>
import { useAnnotationStore } from '@/stores/annotationStore'
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const annotationStore = useAnnotationStore()
const router = useRouter()

const manuscripts = ref([])
const models = ref([])
const manuscript_name = ref()
const model = ref()

const RECOGNITION_URL = import.meta.env.VITE_BACKEND_URL + '/recognise'

function fetch_manuscript() {
  fetch(RECOGNITION_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ manuscript_name: manuscript_name.value, model: model.value }),
  })
    .then((response) => response.json())
    .then((object) => {
      const manuscript_name = Object.values(object)[0][0].manuscript_name
      const selected_model = Object.values(object)[0][0].selected_model
      annotationStore.recognitions[manuscript_name] = {}

      for (const page of Object.keys(object)) {
        annotationStore.recognitions[manuscript_name][page] = {}
        for (const line in object[page]) {
          const line_name = object[page][line]['line']
          annotationStore.recognitions[manuscript_name][page][line_name] = {}
          annotationStore.recognitions[manuscript_name][page][line_name]['predicted_label'] =
            object[page][line]['predicted_label']
          annotationStore.recognitions[manuscript_name][page][line_name]['image_path'] =
            object[page][line]['image_path']
          annotationStore.recognitions[manuscript_name][page][line_name]['confidence_score'] =
            object[page][line]['confidence_score']
        }
      }
      annotationStore.userAnnotations.push({
        manuscript_name: manuscript_name,
        selected_model: selected_model,
        annotations: {},
      })

      router.push({ name: 'annotation-section' })
    })
}

fetch(import.meta.env.VITE_BACKEND_URL + '/uploaded-manuscripts')
  .then((response) => response.json())
  .then((object) => {
    manuscripts.value = object
    manuscript_name.value = manuscripts.value[0]
  })

fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((response) => response.json())
  .then((object) => {
    models.value = object
    model.value = models.value[0]
  })
</script>

<template>
  <div class="uploadedManuscriptsView-container">
    <header>
      <h1>Manuscript Annotation Tool</h1>
    </header>
    <form
      v-if="manuscripts.length && models.length"
      class="mb-3"
      @submit.prevent="fetch_manuscript"
    >
      <label for="page" class="form-label">Manuscript</label>
      <select
        v-model="manuscript_name"
        class="form-select"
        id="page"
        placeholder="Select a manuscript"
      >
        <option v-for="manuscript in manuscripts" :key="manuscript">
          {{ manuscript }}
        </option>
      </select>
      <label for="model" class="form-label">Model</label>
      <select class="form-select mb-3" id="model" v-model="model" placeholder="Select a model">
        <option disabled hidden value="">Select a model</option>
        <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
      </select>
      <button type="submit" class="btn btn-primary">Find</button>
    </form>
  </div>
</template>

<style>
.uploadedManuscriptsView-container {
  padding: 1em;
}
</style>
