

<script setup>
import { ref } from 'vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import AnnotationBlock from './AnnotationBlock.vue'

const props = defineProps(['data', 'page_name', 'manuscript_name'])
const annotationStore = useAnnotationStore()

if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
  annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
}

const selectedIndex = ref(0)

const lines = Object.entries(props.data).map(([line_name, line_data]) => ({
  line_name,
  ...line_data,
}))

function handleNavigate(newIndex) {
  if (newIndex >= 0 && newIndex < lines.length) {
    selectedIndex.value = newIndex
  }
}
</script>

<template>
  <div class="page-view">
    <AnnotationBlock
      v-for="(line, idx) in lines"
      :key="line.line_name"
      :line_name="line.line_name"
      :line_data="{ ...line, selected: selectedIndex === idx }"
      :page_name="props.page_name"
      :manuscript_name="props.manuscript_name"
      :line_index="idx"
      :total_lines="lines.length"
      :onNavigate="handleNavigate"
    />
  </div>
</template>

<style scoped>
.page-view {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;

  gap: 0;
  width: 80%;
  margin-left: 20px;
}
</style>
