<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const manuscripts = ref([])
const pages = ref([])
const selectedManuscript = ref('')
const selectedPage = ref('')
const isLoadingManuscripts = ref(true)
const isLoadingPages = ref(false)

const fetchManuscripts = async () => {
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/uploaded-manuscripts`)
    if (!response.ok) throw new Error('Failed to fetch manuscripts')
    manuscripts.value = await response.json()
  } catch (error) {
    console.error(error)
    alert('Could not load manuscripts.')
  } finally {
    isLoadingManuscripts.value = false
  }
}

const fetchPages = async (manuscriptName) => {
  if (!manuscriptName) return
  isLoadingPages.value = true
  pages.value = []
  selectedPage.value = ''
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/manuscript/${manuscriptName}/pages`
    )
    if (!response.ok) throw new Error('Failed to fetch pages')
    pages.value = await response.json()
  } catch (error) {
    console.error(error)
    alert(`Could not load pages for ${manuscriptName}.`)
  } finally {
    isLoadingPages.value = false
  }
}

onMounted(fetchManuscripts)

watch(selectedManuscript, (newManuscript) => {
  fetchPages(newManuscript)
})

const loadEditor = () => {
  if (selectedManuscript.value && selectedPage.value) {
    router.push({
      name: 'edit-manuscript-layout',
      params: {
        manuscriptName: selectedManuscript.value,
        pageName: selectedPage.value,
      },
    })
  } else {
    alert('Please select a manuscript and a page.')
  }
}
</script>

<template>
  <div class="selection-container p-4">
    <h2>Edit Manuscript Layout</h2>
    <p>Select a manuscript and a page to begin editing the layout graph.</p>

    <div class="mb-3">
      <label for="manuscript-select" class="form-label">Manuscript</label>
      <select
        id="manuscript-select"
        class="form-select"
        v-model="selectedManuscript"
        :disabled="isLoadingManuscripts"
      >
        <option disabled value="">
          {{ isLoadingManuscripts ? 'Loading...' : 'Select a manuscript' }}
        </option>
        <option v-for="ms in manuscripts" :key="ms" :value="ms">{{ ms }}</option>
      </select>
    </div>

    <div class="mb-3" v-if="selectedManuscript">
      <label for="page-select" class="form-label">Page</label>
      <select id="page-select" class="form-select" v-model="selectedPage" :disabled="isLoadingPages">
        <option disabled value="">
          {{ isLoadingPages ? 'Loading pages...' : 'Select a page' }}
        </option>
        <option v-for="page in pages" :key="page" :value="page">{{ page }}</option>
      </select>
    </div>

    <button
      @click="loadEditor"
      class="btn btn-primary"
      :disabled="!selectedManuscript || !selectedPage"
    >
      Load Editor
    </button>
  </div>
</template>

<style scoped>
.selection-container {
  max-width: 600px;
  margin: auto;
}
</style>