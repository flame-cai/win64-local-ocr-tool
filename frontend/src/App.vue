<script setup>
import { RouterView } from 'vue-router'

function setThemeBasedOnPreference() {
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)').matches
  document.documentElement.setAttribute('data-bs-theme', prefersDarkScheme ? 'dark' : 'light')
}

setThemeBasedOnPreference()

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  setThemeBasedOnPreference()
})
</script>

<template>
  <RouterView />
</template>
