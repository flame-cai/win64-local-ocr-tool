<script setup>
import { useAnnotationStore } from '@/stores/annotationStore'
import AnnotationBlock from './AnnotationBlock.vue'

const props = defineProps(['data', 'page_name', 'manuscript_name'])
const annotationStore = useAnnotationStore()

// This initialization is still useful. The check prevents overwriting existing data.
if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
  annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
}
</script>

<template>
  <div>
    <div v-for="(line_data, line_name) in props.data" :key="line_name">
      <AnnotationBlock
        :line_name="line_name"
        :line_data="line_data"
        :page_name="props.page_name"
        :manuscript_name="props.manuscript_name"
      />
    </div>
  </div>
</template>