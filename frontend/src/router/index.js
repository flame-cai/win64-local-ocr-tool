import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'welcome',
      component: () => import('../views/LandingPage.vue'),
    },
    {
      path: '/annotation',
      name: 'annotation-view',
      component: () => import('../views/AnnotationView.vue'),
      children: [
        {
          path: '/annotation/upload',
          name: 'upload-manuscript',
          component: () => import('../components/UploadForm.vue'),
          alias: '/annotation',
        },
        {
          path: '/annotation/annotate',
          name: 'annotation-section',
          component: () => import('../components/AnnotationSection.vue'),
        },
        {
          path: '/annotation/segment',
          name: 'segment',
          component: () => import('../components/SegmentationSection.vue'),
        },
        {
          path: '/annotation/semi-segment',
          name: 'semi-segment',
          component: () => import('../components/SemiSegmentationSection.vue'),
        },
      ],
    },


    {
      path: '/new',
      name: 'new-annotation-view',
      component: () => import('../views/new-AnnotationView.vue'),
      children: [
        {
          path: '/new/upload',
          name: 'new-manuscript',
          component: () => import('../components/new-UploadForm.vue'),
        },
        {
          path: '/new/layout-analysis',
          name: 'layout-analysis',
          component: () => import('../components/new-LayoutAnalysis.vue'),
        }
      ],
    },

    
    {
      path: '/uploads',
      name: 'uploaded-manuscripts',
      component: () => import('../views/UploadedManuscriptsView.vue'),
    },




  ],
})

export default router
