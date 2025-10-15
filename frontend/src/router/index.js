import Dashboard from '@/views/Dashboard.vue'
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'welcome',
      component: () => import('../views/Home.vue'),
    },
    {
      path: '/annotation',
      name: 'annotation-view',
      component: () => import('../views/AnnotationView.vue'),
      children: [
        {
          path: '/annotation/upload',
          name: 'upload-manuscript',
          component: () => import('../components/archive/UploadForm.vue'),
          alias: '/annotation',
        },
        {
          path: '/annotation/annotate',
          name: 'annotation-section',
          component: () => import('../components/archive/AnnotationSection.vue'),
        },
        {
          path: '/annotation/semi-segment',
          name: 'semi-segment',
          component: () => import('../components/archive/SemiSegmentationSection.vue'),
        },
      ],
    },

    {
      path: '/new',
      name: 'new-annotation-view',
      component: () => import('../views/new-AnnotationView.vue'),
      children: [
        {
          path: 'upload',
          name: 'new-manuscript',
          component: () => import('../components/new-UploadForm.vue'),
        },
        {
          path: 'img-2-txt',
          name: 'img-2-txt',
          component: () => import('../components/new-IMG2TXT.vue'),
        },
        {
          path: 'semi-segment',
          name: 'new-semi-segment',
          component: () => import('../components/new-SemiSegmentationSection.vue'),
        },
      ],
    },

    {
      path: '/edit',
      component: () => import('../views/new-AnnotationView.vue'), // Reuse the main view wrapper
      children: [
        {
          path: '', // Default path for /edit
          name: 'edit-manuscript-select',
          component: () => import('../views/EditSelectionView.vue'),
        },
        {
          path: ':manuscriptName/:pageName',
          name: 'edit-manuscript-layout',
          component: () => import('../components/new-SemiSegmentationSection.vue'),
          props: true, // Pass route params (:manuscriptName, :pageName) as props
        },
      ],
    },

    {
      path: '/uploads',
      name: 'uploaded-manuscripts',
      component: () => import('../views/UploadedManuscriptsView.vue'),
    },
    {
      path: '/dashboard',
      name: 'dashboard',
      component: () => import('../views/Dashboard.vue'),
    },
  ],
})
router.beforeEach((to, from, next) => {
  const publicPaths = ['/']
  const authRequired = !publicPaths.includes(to.path)
  const user = localStorage.getItem('user')

  if (authRequired && !user) {
    next('/')
    alert('Please sign in first')
  } else {
    next()
  }
})
export default router
