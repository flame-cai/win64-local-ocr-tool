<script setup>
import { useAnnotationStore } from '@/stores/annotationStore'
import AnnotationBlock from './new-AnnotationBlock.vue'

const props = defineProps(['data', 'page_name', 'manuscript_name'])
const annotationStore = useAnnotationStore()

// PROBLEM AREA: This line runs when AnnotationPage is initialized.
// If we navigate from Page A -> C -> B, and in C we haven't "saved" (triggered recognition) yet,
// then props.data might be an empty object initially.
// More importantly, this line re-initializes the user annotations for the page EVERY time
// AnnotationPage is rendered, potentially wiping out existing user annotations if they switch pages.
// annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
//
// SUGGESTED CHANGE: Initialize only if not already present.
// And ensure there's a userAnnotation entry.

if (annotationStore.userAnnotations.length > 0) {
  if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
    annotationStore.userAnnotations[0]['annotations'][props.page_name] = {};
  }
} else {
  // This case should ideally be handled earlier, e.g., when a manuscript is first processed
  // or when the user logs in/starts a session.
  // For now, we can log a warning or create the initial structure if absolutely necessary,
  // but it's better if userAnnotations[0] is guaranteed to exist by this point.
  console.warn('userAnnotations array is empty. Cannot initialize page annotations.');
  // If you MUST initialize it here (less ideal):
  // annotationStore.userAnnotations.push({
  // manuscript_name: props.manuscript_name, // Or get from store if available
  // selected_model: annotationStore.modelName, // Or get from store
  // annotations: {
  // [props.page_name]: {}
  // }
  // });
}

</script>

<template>
  <div>
    <!-- This v-for loop is fine. It will correctly iterate over props.data
         which comes from annotationStore.recognitions[manuscript_name][page_name].
         If props.data is empty (e.g., before recognition on Page C is done),
         nothing will be rendered here, which is correct. -->
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