<script setup>
import { ref, inject } from "vue"
import { GoogleLogin } from "vue3-google-login"

const axios = inject("axios")
const user = ref(null)
const showDropdown = ref(false)

const handleSuccess = async (response) => {
  try {
    const res = await axios.post("http://localhost:5000/auth/google", {
      code: response.code,
    })
    user.value = res.data.user
    localStorage.setItem("user", JSON.stringify(user.value))
  } catch (err) {
    console.error("❌ Backend error:", err)
  }
}

const handleError = (error) => {
  console.error("Google login failed:", error)
}

const logout = () => {
  user.value = null
  localStorage.removeItem("user")
  showDropdown.value = false
}

const storedUser = localStorage.getItem("user")
if (storedUser) {
  user.value = JSON.parse(storedUser)
}
</script>

<template>
  <header class="header">
    <h1 class="title">Manuscript Annotation Tool</h1>

    <nav class="nav">
      <template v-if="!user">
        <GoogleLogin :callback="handleSuccess" @error="handleError">
          <button class="google-btn">
            <img
              class="google-icon"
              src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
              alt="Google"
            />
            <span>Sign in with Google</span>
          </button>
        </GoogleLogin>
      </template>

      <template v-else>
        <span @click="$router.push('/')" class="home-text">Home</span>
        <button @click="$router.push('/dashboard')" class="dashboard-btn">
          Dashboard
        </button>

        <div class="profile-container" @click="showDropdown = !showDropdown">
          <img :src="user.picture" alt="User" class="avatar" />
          <transition name="fade">
            <div v-if="showDropdown" class="dropdown">
              <img :src="user.picture" alt="User" class="dropdown-avatar" />
              <p class="dropdown-name">{{ user.username }}</p>
              <p class="dropdown-email">{{ user.email }}</p>
              <hr />
              <button @click.stop="logout" class="dropdown-logout">
                Logout
              </button>
            </div>
          </transition>
        </div>
      </template>
    </nav>
  </header>
</template>

<style scoped>
.header {
  background-color: #ffffff;
  padding: 16px 32px;
  display: flex;
  width:100%;
  justify-content: space-between;
  align-items: center;
  border-bottom: rgb(243, 187, 84) 4px solid;
}

.title {
  color: #32363b;
  font-size: 24px;
  font-weight: 700;
   flex-shrink: 0; 
}

.nav {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.google-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  background-color: #ffffff;
  color: #5f6368;
  font-weight: 500;
  font-size: 14px;
  border: 1px solid #dadce0;
  border-radius: 6px;
  padding: 8px 16px;
  cursor: pointer;
  transition: box-shadow 0.2s ease;
}
.google-btn:hover {
  box-shadow: 0 1px 3px rgba(60, 64, 67, 0.3),
    0 4px 8px rgba(60, 64, 67, 0.15);
}
.google-icon {
  width: 20px;
  height: 20px;
}

.profile-container {
  position: relative;
  cursor: pointer;
}
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 2px solid #e5e7eb;
  object-fit: cover;
}
.home-text{
  color: #32363b;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
}
.dropdown {
  position: absolute;
  top: 45px;
  right: 0;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  width: fit-content;
  box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
  text-align: center;
  z-index: 100;
}

.dropdown-avatar {
  width: fit-content;
  height: 50px;
  border-radius: 50%;
  margin-bottom: 8px;
}

.dropdown-name {
  font-weight: 600;
  color: #374151;
  margin: 0;
}

.dropdown-email {
  font-size: 13px;
  color: #6b7280;
  margin: 0 0 10px;
}

.dropdown-logout {
  background-color: #ef4444;
  color: white;
  font-weight: 600;
  border: none;
  padding: 8px;
  border-radius: 6px;
  cursor: pointer;
  width: 100%;
}
.dropdown-logout:hover {
  background-color: #dc2626;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
.dashboard-btn{
  background-color: #2563eb;
  color: white;
  font-weight: 600;
  border: none;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
.dashboard-btn:hover {
  background-color: #1e40af;
}
</style>
