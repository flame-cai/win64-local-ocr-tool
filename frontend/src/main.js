import './assets/main.scss'
import "dropzone/dist/dropzone.css";

import { createApp } from 'vue'
import  { createPinia } from 'pinia'; 
import App from './App.vue'
import router from './router';
import vue3GoogleLogin from 'vue3-google-login'
import axios from 'axios';
const app = createApp(App)
const pinia = createPinia();
app.use(vue3GoogleLogin, {
  clientId: "450300910281-jk68qakom7gkhvgb5ae61ngni5lcg31k.apps.googleusercontent.com"
});

app.provide('axios', axios);

app.use(pinia)
app.use(router)
app.mount('#app')