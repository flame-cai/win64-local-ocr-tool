<script setup>
import { RouterView } from 'vue-router'
import AppHeader from './components/Header.vue'

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
  <AppHeader/>
  <RouterView />
</template>
