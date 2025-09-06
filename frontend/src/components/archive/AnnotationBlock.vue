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
  'mb-2': true,
  'me-2': true,
  'devanagari-textbox': true,
  'is-valid': false,
})

// This computed property is the key. It reads from the store but falls back to the
// original prop for display, and only writes to the store when changed.
const devanagariText = computed({
  get() {
    return annotationStore.userAnnotations[0]?.annotations?.[props.page_name]?.[props.line_name]?.ground_truth ?? props.line_data.predicted_label
  },
  set(newValue) {
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
      annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
    }
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
      annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {}
    }
    
    annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] = newValue
    textboxClassObject['is-valid'] = true
  }
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
  // We only mark as valid if an annotation *already exists* in the store
  // (e.g., from a previous edit in the same session).
  // We no longer pre-populate the store from here.
  if (annotationStore.userAnnotations[0]?.annotations?.[props.page_name]?.[props.line_name]?.ground_truth) {
    textboxClassObject['is-valid'] = true
  }
    
  if (devanagariInput.value) {
    devanagariInput.value.addEventListener('keydown', boundHandleInput)
  }
})
</script>

<template>
  <img
    :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_name}`"
    class="mb-2 manuscript-segment-img"
  />
  <div class="annotation-input">
    <input 
      ref="devanagariInput"
      v-model="devanagariText" 
      type="text" 
      :class="textboxClassObject" 
    />
    <button class="btn btn-primary mb-2 me-2" @click="toggleHK">Roman</button>
  </div>
  <input v-model="hk" type="text" class="form-control mb-2" v-if="isHK" />
</template>

<style>
.manuscript-segment-img {
  display: block;
}

.annotation-input {
  width: 100%;
  display: flex;
}

.devanagari-textbox {
  flex-grow: 1;
  display: inline-block;
}
</style>