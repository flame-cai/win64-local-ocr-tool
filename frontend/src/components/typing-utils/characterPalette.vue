<script setup>
import { ref } from 'vue'

const togglePalette = ref(false)
const copied = ref()

function copyToClipboard(char) {
  navigator.clipboard
    .writeText(char)
    .then(() => {
      copied.value = char; 
    })
    .catch((err) => {
      console.error('Failed to copy:', err)
    })
}

const characters = [
  'ऀ',
  'ऄ',
  'ऎ',
  'ऒ',
  'ऴ',
  'ऺ',
  'ऻ',
  'ऽ',
  'ॆ',
  'ॊ',
  'ॎ',
  'ॏ',
  '॑',
  '॒',
  'ॕ',
  'ॖ',
  'ॗ',
  'ऌ',
  'ॡ',
  'ॢ',
  'ॢ',
  '॥',
  '॰',
  'ॲ',
  'ॳ',
  'ॴ',
  'ॵ',
  'ॶ',
  'ॷ',
  'ॸ',
  'ॹ',
  'ॺ',
  'ॻ',
  'ॼ',
  'ॽ',
  'ॾ',
  'ॿ',
  '꣠',
  '꣡',
  '꣢',
  '꣣',
  '꣤',
  '꣥',
  '꣦',
  '꣧',
  '꣨',
  '꣩',
  '꣪',
  '꣫',
  '꣬',
  '꣭',
  '꣮',
  '꣯',
  '꣰',
  '꣱',
  'ꣲ',
  'ꣳ',
  'ꣴ',
  'ꣵ',
  'ꣶ',
  'ꣷ',
  '꣸',
  '꣹',
  '꣺',
  'ꣻ',
  '꣼',
  'ꣽ',
  'ꣾ',
  'ꣿ',
]
</script>

<template>
  <div class="characterPalette-container">
    <button @click="togglePalette = !togglePalette" class="btn btn-outline-warning me-2">
      ॳ Rare Characters
    </button>
    <span v-if="copied">Copied &zwnj;{{ copied }} !</span>
    <div v-if="togglePalette" class="characterPalette mt-2">
      <button
        class="btn btn-outline-secondary character-button"
        v-for="character in characters"
        :key="character"
        @click="copyToClipboard(character)"
      >
        {{ character }}
      </button>
    </div>
  </div>
</template>

<style>
.characterPalette-container {
  display: inline-block;
  /* max-width: 20rem; */
}

.characterPalette {
  /* display: flex; */
  position: absolute;
  max-width: 20rem;
  /* flex-shrink: 1; */
  padding: 0.5em;
  background-color: var(--color-background);
  border: var(--bs-border-width) solid var(--bs-border-color);
  border-radius: var(--bs-border-radius);
  justify-content: space-around;
}

.character-button {
  width: 2rem;
  margin-left: 0.5em;
  margin-bottom: 0.5em;
  padding: 6px 0px 6px 0px;
  text-align: center;
  background-color: var(--bs-body-bg);
}
</style>
