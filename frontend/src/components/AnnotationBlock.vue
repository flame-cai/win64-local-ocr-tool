<script setup>
import { reactive, ref, watch, onMounted } from 'vue'
import Sanscript from '@indic-transliteration/sanscript'
import { useAnnotationStore } from '@/stores/annotationStore'
import { handleInput } from './typing-utils/devanagariInputUtils'  // Import the new utility function

const BASE_PATH = `${import.meta.env.VITE_BACKEND_URL}/line-images`

const props = defineProps(['line_name', 'line_data', 'page_name', 'manuscript_name'])
const annotationStore = useAnnotationStore()

const isHK = ref(false)

const textboxClassObject = reactive({
  'form-control': true,
  'mb-2': true,
  'me-2': true,
  'devanagari-textbox': true,
  'is-valid': false,
})

const devanagari = ref(props.line_data.predicted_label)
const hk = ref(Sanscript.t(props.line_data.predicted_label, 'devanagari', 'hk'))

const devanagariInput = ref(null)

watch(hk, function () {
  if (!isHK.value) return
  devanagari.value = Sanscript.t(hk.value, 'hk', 'devanagari')
})

function toggleHK() {
  hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk')
  isHK.value = !isHK.value
}

function save() {
  annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {}
  annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name][
    'ground_truth'
  ] = devanagari.value
  textboxClassObject['is-valid'] = true
}


const boundHandleInput = (event) => handleInput(event, devanagari)

onMounted(() => {
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
      v-model="devanagari" 
      type="text" 
      :class="textboxClassObject" 
    />
    <button class="btn btn-primary mb-2 me-2" @click="toggleHK">Roman</button>
    <button class="btn btn-success mb-2 me-2" @click="save">Save</button>
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