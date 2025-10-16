
<script setup>
import { reactive, ref, watch, onMounted, computed, nextTick } from 'vue'
import Sanscript from '@indic-transliteration/sanscript'
import { useAnnotationStore } from '@/stores/annotationStore'
import { handleInput } from '../typing-utils/devanagariInputUtils'

const BASE_PATH = `${import.meta.env.VITE_BACKEND_URL}/line-images`

const props = defineProps([
  'line_name',
  'line_data',
  'page_name',
  'manuscript_name',
  'line_index',
  'total_lines',
  'onNavigate',
])
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

  // Smooth scroll when selected
  if (props.line_data.selected) {
    nextTick(() => {
      devanagariInput.value?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  }
})
</script>

<template>
  <div class="annotation-line">
    <img
      :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_name}`"
      class="line-image"
      alt="Manuscript line"
      :class="{ selected: props.line_data.selected }"
      @click="props.onNavigate(props.line_index)"
    />

    <transition name="fade">
      <div v-if="props.line_data.selected" class="transcription-panel">
        <div class="d-flex align-items-center gap-2">
          <input
            ref="devanagariInput"
            v-model="devanagariText"
            type="text"
            :class="textboxClassObject"
            placeholder="Enter Devanagari text..."
          />

          
          <button
            class="btn btn-outline-secondary"
            :class="{ 'btn-dark': isHK, active: isHK }"
            type="button"
            @click="toggleHK"
          >
            Roman
          </button>
          <button
            class="btn btn-outline-primary"
            :disabled="props.line_index === 0"
            @click="props.onNavigate(props.line_index - 1)"
            title="Go to previous line"
          >
            ↑
          </button>

          <button
            class="btn btn-outline-primary"
            :disabled="props.line_index === props.total_lines - 1"
            @click="props.onNavigate(props.line_index + 1)"
            title="Go to next line"
          >
            ↓
          </button>
        </div>

        <input
          v-if="isHK"
          v-model="hk"
          type="text"
          class="form-control mt-2"
          placeholder="Enter Harvard-Kyoto text..."
        />
      </div>
    </transition>
  </div>
</template>

<style scoped>
.annotation-line {
  display: flex;
  flex-direction: column;
  align-items: left;
  width: 100%;

  border: none;
}

.line-image {
  width: 100%;
  height: 50px;
  cursor: pointer;
  object-fit:contain;
  object-position: left center;         
  border: none;
  transition: transform 0.2s ease, outline 0.2s ease;
}

.line-image.selected {
  outline: 2px solid #007bff;
   margin-left: 20px;
  z-index: 2;
}

.transcription-panel {
  width: 100%;

  background-color: #f8f9fa;
  padding: 0.5rem 0.75rem;
  border-top: 1px solid #ccc;
}

.devanagari-textbox {
  flex: 1;
  width: 100%;
  color: #000;
  font-size: 1.4em;
  line-height: 1.5;
  background-color: #fff;
}

.nav-btn {
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  padding: 0;
}

.btn-outline-secondary.active.btn-dark {
  background-color: var(--bs-dark);
  border-color: var(--bs-dark);
  color: #fff;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

</style>
