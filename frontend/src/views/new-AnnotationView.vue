<script setup>
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="annotationView-container">
    <header class="header">
      <h1>Historical OCR Tool</h1>
    </header>
    <RouterView />
  </div>
</template>

<style>
.annotationView-container {
  padding: 1em;
}

.header {
  padding: 2px 4px;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  width: fit-content;
  border: none;
  border-bottom: 2px solid #333;
  border-radius: 0;
  margin-bottom: 1em;
}

.header h1 {
  font-size: 1.5em;
  color: #333;
  margin: 0;
}
</style>
