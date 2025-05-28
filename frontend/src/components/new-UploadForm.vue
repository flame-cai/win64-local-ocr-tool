<script setup>
import Dropzone from 'dropzone'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnnotationStore } from '@/stores/annotationStore'

const annotationStore = useAnnotationStore()
const uploadForm = ref()
const manuscriptName = ref('') // Initialize to prevent undefined issues, bind with v-model
const models = ref([])
const modelSelected = ref('') // Initialize, bind with v-model

const router = useRouter()

fetch(import.meta.env.VITE_BACKEND_URL + '/models')
  .then((response) => response.json())
  .then((object) => {
    models.value = object
  })

onMounted(() => {
  uploadForm.value = new Dropzone('#upload-form', {
    url: import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript', // Set URL here for Dropzone
    uploadMultiple: true,
    autoProcessQueue: false, // We'll call processQueue manually
    parallelUploads: Infinity,
    // It's good practice to define acceptedFiles if you expect specific image types
    // acceptedFiles: 'image/jpeg,image/png,image/gif', // Example
  })

  uploadForm.value.on('completemultiple', function (files) {
    // This event fires after all files in a batch are processed (uploaded or failed).
    // We should ideally check file.status === Dropzone.SUCCESS for each file,
    // but for simplicity, we'll assume 'files' here are the ones intended for processing if the overall batch didn't error out.
    // If Dropzone's 'successmultiple' event is available and preferred, that could be used too.
    

    const currentManuscriptNameFromForm = manuscriptName.value
    const currentModelSelectedFromForm = modelSelected.value
    console.log('Page A: currentModelSelectedFromForm is:', currentModelSelectedFromForm); // DEBUG
    console.log('Page A: manuscriptName.value is:', manuscriptName.value); // DEBUG

    // 1. Basic validation: Ensure manuscript name and model are selected
    if (!currentManuscriptNameFromForm) {
      alert('Please enter a manuscript name.')
      console.error('Manuscript name not provided.')
      // Potentially clear the queue or re-enable the form
      // uploadForm.value.removeAllFiles(); // if you want to clear on error
      return
    }
    if (!currentModelSelectedFromForm) {
      alert('Please select a model.')
      console.error('Model not selected.')
      // uploadForm.value.removeAllFiles();
      return
    }
    if (files.length === 0) {
      alert('Please add files to upload.')
      console.error('No files in the upload queue for processing.')
      return
    }

    console.log('Upload complete. Backend responded. Now updating store from frontend data.')

    // 2. Update annotationStore
    annotationStore.reset(); // Optional: Reset store if starting a new manuscript processing session

    // Set the model name
    annotationStore.modelName = currentModelSelectedFromForm

    // Initialize recognitions for this manuscript
    annotationStore.recognitions[currentManuscriptNameFromForm] = {}

    // 3. Populate annotationStore.recognitions using uploaded file names as page IDs
    // Extract successfully uploaded file names. Dropzone's `files` in `completemultiple`
    // should be the list of files processed in that batch.
    const uploadedPageIds = files
      .filter(file => file.status === Dropzone.SUCCESS && file.name) // Ensure successful & has a name
      .map(file => file.name.split('.')[0]) // Use the original filename as the page ID
      .filter(pageId => pageId && pageId.trim() !== '')

    if (uploadedPageIds.length === 0 && files.length > 0) {
        console.warn("No files were successfully uploaded, or they lack names. Cannot set up pages.");
        // Potentially inform user more directly
        return;
    }
    if (uploadedPageIds.length === 0 && files.length === 0) {
        console.log("No files were in the queue to begin with.");
        return;
    }


    // Sort page IDs (filenames) to ensure consistent order
    // The store's `sortedPageIds` computed property will also sort,
    // but sorting here before insertion might be slightly cleaner if order of keys matters for non-display logic.
    // However, relying on `sortedPageIds` from the store is the canonical way.
    // For now, we just add them, and the store's computed property will handle sorting for navigation.
    uploadedPageIds.forEach(pageId => {
      // For each uploaded file, create an entry in recognitions.
      // Initially, there's no line data (predicted_label, etc.) because
      // the backend response is just "Hello world".
      // This structure prepares the store for these pages.
      // Line data would need to be fetched later or entered manually.
      annotationStore.recognitions[currentManuscriptNameFromForm][pageId] = {}
    })

    // 4. Update userAnnotations array
    annotationStore.userAnnotations.push({
      manuscript_name: currentManuscriptNameFromForm,
      selected_model: currentModelSelectedFromForm,
      annotations: {}, // User's ground truth annotations will go here
    })

    // 5. Set the initial page in the store
    // This will use the `sortedPageIds` computed property, which now knows about the pages
    // we just added from the uploaded filenames.
    annotationStore.setInitialPage()

    if (!annotationStore.currentPage) {
        console.warn("Current page could not be set. This might happen if no files were successfully processed as pages.");
        // Handle this case, maybe route to a different page or show an error.
    }

    // 6. Navigate to the annotation view
    console.log(`Navigating. Current page set to: ${annotationStore.currentPage}`)
    router.push({ name: 'new-semi-segment' })
  })

  // It's important that Dropzone knows where to send the files.
  // The <form :action="UPLOAD_URL"> is for non-JS submissions.
  // Dropzone needs its own `url` option, or it will use the form's action.
  // Ensure your Dropzone instance is configured with the correct upload URL.
  // If '#upload-form' already has an action attribute, Dropzone might pick it up.
  // Explicitly setting it in Dropzone options is safer:
  // new Dropzone('#upload-form', { url: UPLOAD_URL, ... })
  // The code above already includes it.
})

// UPLOAD_URL is used by the button click handler with processQueue, Dropzone uses its 'url' option.
// const UPLOAD_URL = import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript'; // Already defined
</script>

<template>
  <div class="mb-3">
    <label for="manuscriptName" class="form-label">Manuscript Name</label>
    <input type="text" class="form-control" id="manuscriptName" v-model="manuscriptName" />
  </div>
  <div class="mb-3">
    <label for="model" class="form-label">Model</label>
    <select class="form-select" id="model" v-model="modelSelected"> <!-- Removed placeholder for v-model -->
      <option disabled value="">Select a model</option> <!-- Default disabled option -->
      <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
    </select>
  </div>
  <!--
    The action attribute on the form is a fallback or can be used by Dropzone if `url` option isn't set.
    For Dropzone, the `url` option specified during initialization is primary.
    The hidden inputs for manuscript_name and model are still useful as Dropzone will include them as form data.
  -->
  <form :action="UPLOAD_URL" class="dropzone" id="upload-form">
    <div class="dz-message" data-dz-message><span>Drop files here or click to upload.</span></div>
    <div class="previews"></div> <!-- Dropzone might use this for previews if configured -->
    <input type="hidden" name="manuscript_name" :value="manuscriptName" />
    <input type="hidden" name="model" :value="modelSelected" />
  </form>
  <button @click="uploadForm.processQueue()" class="btn btn-primary mt-3"
          :disabled="!manuscriptName || !modelSelected"> <!-- Disable button if critical info missing -->
    Submit
  </button>
</template>

<style>
form.dropzone {
  background-color: var(--bs-body-bg);
  border: var(--bs-border-width) solid var(--bs-border-color);
  color: var(--bs-body-color);
  font-family: inherit;
  min-height: 150px; /* Give dropzone some default height */
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

/* Basic styling for Dropzone message if you use the default one */
.dz-message {
  text-align: center;
  margin: 2em 0;
}
</style>