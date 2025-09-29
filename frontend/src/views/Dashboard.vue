
<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goToNewManuscript = () => router.push("/new/upload")
const goToEditManuscript = () => router.push("/edit")
const goToUploadManuscript = () => router.push("/uploads")
</script>

<template>
  <div class="main-container">
    <div class="header-section">
      <h1>Your Manuscripts</h1>
      <button class="action-btn primary-btn" @click="goToNewManuscript">
        + New Manuscript
      </button>
    </div>

    <div class="manuscript-list-grid">
      <div class="manuscript-card">
        <div class="manuscript-info">
          <p class="manuscript-name">Research Paper Draft 1.0</p>
          <p class="manuscript-detail">Model: LLM v3.1</p>
          <p class="manuscript-detail">Date: 2025-09-25</p>
        </div>
        <div class="manuscript-actions">
          <button class="action-btn" @click="goToEditManuscript">
            Edit Manuscript
          </button>
          <button class="action-btn secondary-btn" @click="goToUploadManuscript">
            Annotate Text
          </button>
        </div>
      </div>
      <div class="manuscript-card">
        <div class="manuscript-info">
          <p class="manuscript-name">Thesis Proposal - Chapter 2</p>
          <p class="manuscript-detail">Model: GPT-4</p>
          <p class="manuscript-detail">Date: 2025-09-20</p>
        </div>
        <div class="manuscript-actions">
          <button class="action-btn" @click="goToEditManuscript">
            Edit Manuscript
          </button>
          <button class="action-btn secondary-btn" @click="goToUploadManuscript">
            Annotate Text</button>
        </div>
      </div>
      <div class="manuscript-card">
        <div class="manuscript-info">
          <p class="manuscript-name">Historical Analysis - Draft</p>
          <p class="manuscript-detail">Model: LLM v2.5</p>
          <p class="manuscript-detail">Date: 2025-08-15</p>
        </div>
        <div class="manuscript-actions">
          <button class="action-btn" @click="goToEditManuscript">
            Edit Manuscript
          </button>
          <button class="action-btn secondary-btn" @click="goToUploadManuscript">
            Annotate Text
          </button>
        </div>
      </div>
      <div class="manuscript-card">
        <div class="manuscript-info">
          <p class="manuscript-name">Final Novel Outline</p>
          <p class="manuscript-detail">Model: Custom</p>
          <p class="manuscript-detail">Date: 2025-07-01</p>
        </div>
        <div class="manuscript-actions">
          <button class="action-btn" @click="goToEditManuscript">
            Edit Manuscript
          </button>
          <button class="action-btn secondary-btn" @click="goToUploadManuscript">
            Annotate Text
          </button>
        </div>
      </div>
      </div>
  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px;
  min-height: 100vh;
  width: 100%;
  background-color: #f7f9fc;
}

.header-section {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
  padding: 0 32px; 
  box-sizing: border-box; 
}

.header-section h1 {
  font-size: 1.8em;
  color: #333;
  text-transform: capitalize;
  margin: 0;
}

.action-btn {
  cursor: pointer;
  padding: 8px 16px;
  font-size: 1em;
  font-weight: 500;
  border-radius: 6px;
  transition: all 0.2s ease;
  margin-left: 10px;
  white-space: nowrap;
}

.primary-btn {
  color: white;
  background-color: #4caf50;
  border: 1px solid #4caf50;
}

.primary-btn:hover {
  background-color: #45a049;
  border-color: #45a049;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transform: translateY(-1px);
}

.secondary-btn {
  color: #4caf50;
  background-color: transparent;
  border: 1px solid #4caf50;
}

.secondary-btn:hover {
  background-color: #e8f5e9;
}

.manuscript-list-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(4, 1fr); 
  gap: 20px;
  padding: 0 32px; 
  box-sizing: border-box;
}

.manuscript-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 20px;
  background-color: white;
  border: 1px solid #ddd;
  border-top: 5px solid #4caf50;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.manuscript-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    cursor: pointer;
}

.manuscript-info {
  flex-direction: column;
  gap: 8px;
  font-size: 0.9em;
  color: #555;
  margin-bottom: 20px;
}

.manuscript-name {
  font-weight: 700;
  color: #222;
  font-size: 1.2em;
  margin: 0 0 10px 0;
}

.manuscript-detail {
  margin: 0;
}

.manuscript-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.manuscript-card .action-btn {
    margin-left: 0;
}


@media (max-width: 1400px) {
  .manuscript-list-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 1024px) {
  .manuscript-list-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 600px) {
  .header-section {
    flex-direction: column;
    align-items: flex-start;
    padding: 0 16px; 
  }
  
  .header-section h1 {
    margin-bottom: 10px;
  }

  .manuscript-list-grid {
    grid-template-columns: 1fr; 
    gap: 15px;
    padding: 0 16px; 
  }

  .manuscript-actions {
    justify-content: space-between;
  }
}
</style>