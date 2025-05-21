<script setup>
import { useAnnotationStore } from '@/stores/annotationStore';
import { useCssModule } from 'vue'
import { RouterLink } from 'vue-router'
const landingPage = useCssModule()
const annotationStore = useAnnotationStore();
annotationStore.reset()
</script>

<template>
  <div :class="landingPage['landing-container']">
    <header>
      <img src="/flame-logo.svg" alt="Flame logo" :class="landingPage.logo" />
      <h1>Manuscript Annotation Tool</h1>
    </header>
    <main :class="landingPage.main">
      <div :class="landingPage.links">
        <RouterLink :to="{ name: 'upload-manuscript' }" class="btn btn-primary m-2"
          >Annotate</RouterLink
        >
        <RouterLink :to="{ name: 'uploaded-manuscripts' }" class="btn btn-primary m-2"
          >Uploaded Manuscripts</RouterLink
        >
      </div>
    </main>
  </div>
</template>

<style module>

.landing-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
}

.logo {
  height: 35vh;
  display: block;
  margin: auto;
}
</style>
