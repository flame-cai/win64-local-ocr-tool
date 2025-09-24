<template>
  <header class="header">
    <h1 class="title">Manuscript Annotation Tool</h1>
    <nav class="nav">
      <span @click="$router.push('/')" class="home-text">Home</span>
      <button @click="$router.push('/dashboard')" class="dashboard-btn">Dashboard</button>
    </nav>
  </header>
</template>

<script>
export default {
  name: "AppHeader"
};
</script>

<style scoped>
.header {
  background-color: #ffffff;
  padding: 16px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  border-bottom: 2px solid #e5e7eb;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
}

.title {
  color: #32363b;
  font-size: 28px;
  font-weight: 700;
  margin: 0;
  letter-spacing: 0.5px;
}

.nav {
  display: flex;
  align-items: center;
  gap: 24px; 
}

.home-text {
  color: #3b82f6;
  font-weight: 600;
  font-size: 18px;
  cursor: pointer;
  transition: color 0.3s ease;
}

.home-text:hover {
  color: #1e40af;
  text-decoration: underline;
}

.dashboard-btn {
  background-color: #2563eb;
  color: white;
  font-weight: 600;
  font-size: 16px;
  border: none;
  padding: 8px 20px;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.dashboard-btn:hover {
  background-color: #1e40af;
  transform: translateY(-2px);
}
</style>
