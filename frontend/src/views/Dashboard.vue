<script setup>
import { useRouter } from 'vue-router' 

const router = useRouter() 

const goToNewManuscript = () => router.push("/new/upload")
const goToEditManuscript = () => router.push("/edit")
const goToUploadManuscript = () => router.push("/uploads") 
</script>

<template>
  <div class="main">
    <button class="new-manuscript-btn" @click="goToNewManuscript">+ New Manuscript</button>

    <button class="new-manuscript-btn" @click="goToUploadManuscript">Annotate Text</button>
    <button class="new-manuscript-btn" @click="goToEditManuscript">Edit Manuscript</button>
  </div>
</template>

<style scoped>
.main {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px;

  min-height: 100vh;
  width: 100%;
  background-color: #f7f9fc;
}

.new-manuscript-btn {
  cursor: pointer;
  padding: 6px 12px;
  font-size: 1.1em;
  font-weight: bold;
  margin: 8px;

  color: white;
  background-color: #4caf50;
  border: 2px solid #4caf50;
  border-radius: 6px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

  margin-left: auto;

  transition: all 0.3s ease;
}

.new-manuscript-btn:hover {
  background-color: #45a049;
  border-color: #45a049;
  box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}
</style>
