<script setup>
import { onMounted, useTemplateRef } from 'vue'

const props = defineProps(['coordinates', 'isSelectMode', 'brushSegment', 'brushColor', 'index'])
const emit = defineEmits(['selected'])
const point = useTemplateRef('point')

function select() {
  if (props.isSelectMode) {
    point.value.style.backgroundColor = props.brushColor
    emit('selected', { segment: props.brushSegment, index: props.index })
  }
}

onMounted(() => {
  point.value.style.left = props.coordinates[0] + 'px'
  point.value.style.top = props.coordinates[1] + 'px'
})
</script>

<template>
  <div ref="point" class="point" @mouseenter="select"></div>
</template>

<style>
.point {
  display: inline-block;
  position: absolute;
  background-color: gray;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.point::before {
  content: '';
  position: absolute;
  top: -5px;
  bottom: -5px;
  left: -5px;
  right: -5px;
  background: transparent;
}
</style>
