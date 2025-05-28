<script setup>
import { reactive, ref, watch, onMounted, computed } from 'vue' // Added computed
import Sanscript from '@indic-transliteration/sanscript'
import { useAnnotationStore } from '@/stores/annotationStore'
import { handleInput } from './typing-utils/devanagariInputUtils'

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

// PROBLEM AREA: props.line_data.predicted_label might not exist if line_data is empty
// (e.g., if recognitions haven't happened yet for this line).
// This can happen if AnnotationPage renders an AnnotationBlock for a line_name
// that exists as a key in props.data, but its value (line_data) is an empty object {}.
//
// The current setup on Page A:
// annotationStore.recognitions[currentManuscriptNameFromForm][pageId] = {}
// This means props.data on AnnotationPage will be an object of pageIds, and the value for each pageId is {}.
// So, when AnnotationPage does v-for over this, line_data will be {} initially.
//
// Then on Page C (saveModifications):
// annotationStore.recognitions[manuscriptName.value][currentPage.value] = responseData.lines;
// This REPLACES the {} with actual line data.
//
// So, the scenario where line_data.predicted_label is undefined is less likely
// IF AnnotationPage only renders blocks when props.data (which is a page's lines) actually has line entries.
// Your AnnotationPage.vue: `v-for="(line_data, line_name) in props.data"`
// If props.data is initially `{"page1": {}}` from Page A, and then becomes `{"page1": {"line_001": {...}, "line_002": {...}}}` after Page C's save,
// then the v-for in AnnotationPage will only run when `props.data` (which is `page_data`) has actual line entries.
// So `props.line_data` inside AnnotationBlock should always be a populated object with `predicted_label`.
//
// Let's assume props.line_data.predicted_label WILL exist if this component is rendered.
// If it could be undefined, we'd need to add checks:
// const initialPredictedLabel = computed(() => props.line_data?.predicted_label || '');
// const devanagari = ref(initialPredictedLabel.value);
// const hk = ref(Sanscript.t(initialPredictedLabel.value, 'devanagari', 'hk'));

const devanagari = ref(props.line_data.predicted_label)
const hk = ref(Sanscript.t(props.line_data.predicted_label, 'devanagari', 'hk'))


const devanagariInput = ref(null)

// Watcher for devanagari changes to update user annotations in real-time (or on blur)
// This is optional, the save button already does this explicitly.
// watch(devanagari, (newValue) => {
//   // Debounce this if you want real-time saving to avoid too many store updates
//   if (annotationStore.userAnnotations.length > 0 &&
//       annotationStore.userAnnotations[0]['annotations'][props.page_name] &&
//       annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
//     annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] = newValue;
//   } else {
//     // Structure might not be ready, defer to save button or ensure structure
//     // is built when line_data is first available.
//   }
// });

watch(hk, function () {
  if (!isHK.value) return
  devanagari.value = Sanscript.t(hk.value, 'hk', 'devanagari')
})

function toggleHK() {
  hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk')
  isHK.value = !isHK.value
}

function save() {
  // Ensure the path in userAnnotations exists before trying to assign to it.
  // AnnotationPage.vue should have initialized userAnnotations[0]['annotations'][props.page_name] = {}
  if (annotationStore.userAnnotations.length > 0 &&
      annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
    // Initialize the specific line's annotation object if it doesn't exist
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
      annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {};
    }
    annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] = devanagari.value;
    textboxClassObject['is-valid'] = true;
  } else {
    console.error("Cannot save annotation, userAnnotations structure not properly initialized for page:", props.page_name);
    // Optionally provide user feedback
  }
}


const boundHandleInput = (event) => handleInput(event, devanagari)

onMounted(() => {
  if (devanagariInput.value) {
    devanagariInput.value.addEventListener('keydown', boundHandleInput);
  }
  // When the component mounts, check if there's already a ground_truth for this line
  // and pre-fill the devanagari input if so. This allows edits to persist across page navigations.
  if (annotationStore.userAnnotations.length > 0 &&
      annotationStore.userAnnotations[0]['annotations'][props.page_name] &&
      annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] &&
      annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] !== undefined) {
    devanagari.value = annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'];
    // Also update hk if devanagari was loaded from store
    hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk');
    textboxClassObject['is-valid'] = true; // Mark as valid if previously saved
  } else {
    // If no existing ground_truth, devanagari.value remains the predicted_label
    // And textboxClassObject['is-valid'] remains false until saved.
  }
})
</script>

<template>
  <!-- The image_path prop for the line is passed as props.line_data.image_path -->
  <!-- In your new backend, props.line_data.image_path IS props.line_name (without extension) -->
  <!-- So, this looks correct. -->
  <img
    :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_data.image_path}`"
    class="mb-2 manuscript-segment-img"
    :alt="`Line image for ${props.line_name}`"
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