<script setup>
import { useCssModule } from 'vue'
import { useRouter } from 'vue-router'

const Home = useCssModule()
const router = useRouter()

const goToDashboard = () =>
{
  const user = localStorage.getItem('user')
  if (!user) return alert('Please sign in first')
  router.push('/dashboard')}
const learnMore = () => alert('Learn more clicked!')
</script>

<template>
  <div :class="Home['landing-container']">
    <div :class="Home['content-row']">
      <div :class="Home['text-and-buttons-section']">
        <img src="/flame-logo.svg" alt="Flame logo" :class="Home.logo" />
        <h1 :class="Home.headline">Manuscript Annotation Tool</h1>
        
        <div :class="Home['buttons-section']">
          <button class="btn btn-primary" @click="goToDashboard">Dashboard</button>
          <button class="btn btn-secondary" @click="learnMore">Learn More</button>
        </div>
      </div>

      <div :class="Home['images-section']">
        <img src="/img1.jpg" alt="Manuscript 1" :class="[Home['manuscript-img'], Home['img1']]" />
        <img src="/img2.jpg" alt="Manuscript 2" :class="[Home['manuscript-img'], Home['img2']]" />
      </div>
    </div>
  </div>
</template>

<style module>
.landing-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  background: #f3f4f6;
  font-family: 'Inter', sans-serif;
  color: #1f2937;
  height : 100vh;
}

.content-row {
  display: flex;
  align-items: center;
  gap: 6rem;
  flex-wrap: wrap;
  max-width: 1200px;
  width: 100%;
}

.text-and-buttons-section {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 1.5rem;
  flex: 1;
}

.logo {
  width: 300px;
  height: auto;
}

.headline {
  font-size: 2.5rem;
  font-weight: 800;
  line-height: 1.2;
  margin: 0;
}

.description {
  font-size: 1.125rem;
  line-height: 1.6;
  color: #4b5563;
  margin: 0;
}

/* Buttons */
.buttons-section {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.btn-primary, .btn-secondary {
  padding: 0.8rem 2rem;
  border: none;
  border-radius: 9999px;
  cursor: pointer;
  font-weight: 600;
  transition: transform 0.2s ease, background-color 0.2s ease;
}

.btn-primary {
  background-color: #1f2937;
  color: white;
}

.btn-primary:hover {
  background-color: #374151;
  transform: translateY(-2px);
}

.btn-secondary {
  background-color: transparent;
  color: #1f2937;
  border: 2px solid #e5e7eb;
}

.btn-secondary:hover {
  border-color: #d1d5db;
  transform: translateY(-2px);
}

/* Right side: Manuscript Images */
.images-section {
  position: relative;
  width: 800px;
  min-height: 500px;
  flex: 1;
}

.manuscript-img {
  width: 100%;
  max-width: 800px;
  height: auto;
  border-radius: 12px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.img1 {
  position: relative;
  z-index: 2;
  top: 0;
  left: 0;
}

.img2 {
  position: absolute;
  left: 100px;
  top: 50px;
  z-index: 1;
}

@media (max-width: 768px) {
  .content-row {
    flex-direction: column;
    text-align: center;
    gap: 2rem;
  }

  .text-and-buttons-section {
    align-items: center;
  }

  .images-section {
    min-height: 300px;
    width: 100%;
  }

  .img2 {
    left: 50px;
    top: 30px;
  }
}
</style>