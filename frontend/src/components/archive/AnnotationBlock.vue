<script setup>
import { reactive, ref, watch, onMounted, computed } from 'vue'
import Sanscript from '@indic-transliteration/sanscript'
import { useAnnotationStore } from '@/stores/annotationStore'
import { handleInput } from '../typing-utils/devanagariInputUtils'

const BASE_PATH = `${import.meta.env.VITE_BACKEND_URL}/line-images`

const props = defineProps(['line_name', 'line_data', 'page_name', 'manuscript_name'])
const annotationStore = useAnnotationStore()

const isHK = ref(false)
const devanagariInput = ref(null)

const textboxClassObject = reactive({
  'form-control': true,
  'devanagari-textbox': true,
  'is-valid': false,
})

const devanagariText = computed({
  get() {
    return (
      annotationStore.userAnnotations[0]?.annotations?.[props.page_name]?.[props.line_name]
        ?.ground_truth ?? props.line_data.predicted_label
    )
  },
  set(newValue) {
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
      annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
    }
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
      annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {}
    }

    annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name][
      'ground_truth'
    ] = newValue
    textboxClassObject['is-valid'] = true
  },
})

const hk = ref(Sanscript.t(devanagariText.value, 'devanagari', 'hk'))

watch(devanagariText, (newValue) => {
  if (!isHK.value) {
    hk.value = Sanscript.t(newValue, 'devanagari', 'hk')
  }
})

watch(hk, (newValue) => {
  if (isHK.value) {
    devanagariText.value = Sanscript.t(newValue, 'hk', 'devanagari')
  }
})

function toggleHK() {
  isHK.value = !isHK.value
  if (isHK.value) {
    hk.value = Sanscript.t(devanagariText.value, 'devanagari', 'hk')
  }
}

const boundHandleInput = (event) => handleInput(event, devanagariText)

onMounted(() => {
  if (
    annotationStore.userAnnotations[0]?.annotations?.[props.page_name]?.[props.line_name]
      ?.ground_truth
  ) {
    textboxClassObject['is-valid'] = true
  }

  if (devanagariInput.value) {
    devanagariInput.value.addEventListener('keydown', boundHandleInput)
  }
})
</script>

<template>
  <div class="annotation-card">
    <div class="card-image-wrapper">
      <img
        :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_name}`"
        class="card-img-top"
        alt="Manuscript line segment"
      />
    </div>

    <div class="card-body">
      <div class="d-flex align-items-center">
        <input
          ref="devanagariInput"
          v-model="devanagariText"
          type="text"
          :class="textboxClassObject"
          placeholder="Enter Devanagari text..."
        />
        <button
          class="btn btn-outline-secondary ms-2"
          :class="{ 'btn-dark': isHK, active: isHK }"
          type="button"
          @click="toggleHK"
        >
          Roman
        </button>
      </div>

      <input
        v-if="isHK"
        v-model="hk"
        type="text"
        class="form-control mt-1"
        placeholder="Enter Harvard-Kyoto text..."
      />
    </div>
  </div>
</template>

<style scoped>
.annotation-card {
  background-color: #f0f0f0;
  border: 1px solid #000;
  border-radius: 0.5rem;
  margin: 1rem auto;
  overflow: hidden;
  width: 80%;
  padding: 4px;
}
.card-image-wrapper {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  background-color: transparent;
}

.card-img-top {
  display: block;
  height: 2.2em;
  width: auto;
  max-width: 100%;
  object-fit: contain;
  object-position: left center;
  max-height: 40px;
}

.card-body {
  padding: 0;
}

.form-control {
  font-size: 1.1em;
  line-height: 1.5;
}

.devanagari-textbox {
  flex-grow: 1;
}

.btn-outline-secondary.active.btn-dark {
  background-color: var(--bs-dark);
  border-color: var(--bs-dark);
  color: #fff;
}

.devanagari-textbox.is-valid {
  border-color: #198754;
  box-shadow: 0 0 0 0.25rem rgba(25, 135, 84, 0.25);
}
</style>
