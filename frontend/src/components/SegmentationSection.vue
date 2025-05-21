<script setup>
import { useAnnotationStore } from '@/stores/annotationStore'
import { onUnmounted, ref, useTemplateRef } from 'vue'
import SegmentationPoint from './SegmentationPoint.vue'
import { useRouter } from 'vue-router'

const annotationStore = useAnnotationStore()
const router = useRouter()
const segmentationCanvas = useTemplateRef('segmentationCanvas')
const points = ref([])
// const pointRefs = useTemplateRef('points');
const isSelectMode = ref(false)
const segments = ref([{ segment: 'segment ' + 1, color: getRandomHexColor() }])
const currentSegment = ref(0)

//need to make this work for different screen sizes
function updateCanvasSize(width, height) {
  segmentationCanvas.value.style.width = width / 2 + 'px'
  segmentationCanvas.value.style.height = height / 2 + 'px'
}

function getRandomHexColor() {
  return `#${Math.floor(Math.random() * 16777215)
    .toString(16)
    .padStart(6, '0')}`
}

function selectPoint(event) {
  const point = event.index
  const segment = event.segment
  console.log(event, point, segment)
  points.value[point]['segment'] = segment
}

function keyUpCanvas() {
  isSelectMode.value = false
  if (currentSegment.value === segments.value.length - 1) {
    segments.value.push({
      segment: 'segment ' + (segments.value.length + 1),
      color: getRandomHexColor(),
      points: [],
    })
    currentSegment.value += 1
  }
}


fetch(
  import.meta.env.VITE_BACKEND_URL +
  `/segment/${Object.keys(annotationStore.recognitions)[0]}/${annotationStore.currentPage}`,
)
.then((response) => response.json())
.then((object) => {
    updateCanvasSize(object['dimensions'][0], object['dimensions'][1])
    object['points'].forEach((point) => {
      points.value.push({
        coordinates: [point[0], point[1]],
        segment: null,
      })
    })
  })

  function createSegments() {
  for (const point of points.value) {
    if (point.segment === null) {
      alert("Some points haven't been assigned a segment")
      return
    }
  }
  const request = points.value.map((point) => point.segment)
  fetch(
    import.meta.env.VITE_BACKEND_URL +
    `/segment/${Object.keys(annotationStore.recognitions)[0]}/${annotationStore.currentPage}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    },
  ).then(() => {
    router.push({ name: 'uploaded-manuscripts' })
  })
}

function onSelectMode(e) {
  if (e.key === 'd')
  isSelectMode.value = true;
}

function offSelectMode(e) {
  if (e.key === 'd')
  {isSelectMode.value = false
  if (currentSegment.value === segments.value.length - 1) {
    segments.value.push({
      segment: 'segment ' + (segments.value.length + 1),
      color: getRandomHexColor(),
      points: [],
    })
    currentSegment.value += 1
  }}
}

document.addEventListener('keydown', onSelectMode)
document.addEventListener('keyup', offSelectMode)
window.addEventListener('blur', offSelectMode)

onUnmounted(() => {
  document.removeEventListener('keyup', onSelectMode)
  document.removeEventListener('mouseup', offSelectMode)
  window.removeEventListener('blur', offSelectMode)
})
</script>

<template>
  <div class="mb-3">
    <button
      class="btn btn-primary me-2"
      @click="
        segments.push({
          segment: 'segment ' + (segments.length + 1),
          color: getRandomHexColor(),
          points: [],
        })
      "
    >
      Add Segment
    </button>
    <button class="btn btn-success" @click="createSegments">Create Segments</button>
    <div class="mb-3">
      <label for="line" class="form-label">Line</label>
      <select class="form-select" v-model="currentSegment">
        <option v-for="(segment, index) in segments" :key="index" :value="index">
          {{ segment.segment }}
        </option>
      </select>
    </div>
  </div>
  <div class="segmentation-container">
    <div
      class="segmentation-canvas"
      ref="segmentationCanvas"
      @keydown.d="isSelectMode = true"
      @keyup.d="keyUpCanvas"
    >
      <SegmentationPoint
        v-for="(point, index) in points"
        :key="index"
        :index="index"
        :coordinates="point.coordinates"
        :isSelectMode="isSelectMode"
        :brushSegment="currentSegment"
        :brushColor="segments[currentSegment]['color']"
        @selected="selectPoint"
      />
    </div>
  </div>
</template>

<style>
.segmentation-container {
  background-color: var(--bs-body-bg);
  border: var(--bs-border-width) solid var(--bs-border-color);
  border-radius: 0.5em;
  padding: 0.5em;
}

.segmentation-canvas {
  background-color: black;
  position: relative;
}
</style>
