//annotationStore.js
import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref, computed as vueComputed } from 'vue' // Use vueComputed to avoid naming clash
import * as zip from "@zip.js/zip.js";

export const useAnnotationStore = defineStore('annotations', () => {
  const modelName = ref();
  const recognitions = ref({}); // Structure: { manuscriptName: { pageId: pageData, ... } }
  const userAnnotations = ref([]);
  const currentPage = ref(); // Stores the ID of the current page, e.g., "001"

  // Helper to get the current (assumed single) manuscript name
  const currentManuscriptName = vueComputed(() => {
    const keys = Object.keys(recognitions.value);
    return keys.length > 0 ? keys[0] : null;
  });

  // Helper to get sorted page IDs for the current manuscript
  const sortedPageIds = vueComputed(() => {
    const manuscript = currentManuscriptName.value;
    // Ensure recognitions.value[manuscript] exists and is an object before trying to get keys
    if (manuscript && recognitions.value[manuscript] && typeof recognitions.value[manuscript] === 'object') {
      return Object.keys(recognitions.value[manuscript]).sort((a, b) => {
        const numA = parseInt(a, 10);
        const numB = parseInt(b, 10);
        // If both are parseable as numbers, sort numerically
        if (!isNaN(numA) && !isNaN(numB)) {
          return numA - numB;
        }
        // Otherwise, fall back to lexicographical sort (e.g., for "page1a", "page1b")
        return a.localeCompare(b);
      });
    }
    return [];
  });

  /**
   * Sets the current page.
   * @param {string} pageId - The ID of the page to set as current.
   */
  function setCurrentPage(pageId) {
    if (currentPage.value !== pageId) {
      console.log(`AnnotationStore: Setting current page to ${pageId}`);
      currentPage.value = pageId;
    }
  }

  /**
   * Navigates to the next page in the current manuscript.
   */
  function nextPage() {
    const pages = sortedPageIds.value;
    
    if (pages.length === 0) {
      console.log("AnnotationStore: No pages available to navigate.");
      return;
    }

    if (!currentPage.value) {
        // If current page is not set, default to the first page.
        console.log("AnnotationStore: Current page not set, navigating to the first available page.");
        setCurrentPage(pages[0]);
        return;
    }

    const currentIndex = pages.indexOf(currentPage.value);
    if (currentIndex === -1) {
      console.warn(`AnnotationStore: Current page "${currentPage.value}" not found in available pages. Navigating to the first page.`);
      setCurrentPage(pages[0]); // Default to first page if current is invalid
      return;
    }

    if (currentIndex < pages.length - 1) {
      setCurrentPage(pages[currentIndex + 1]);
    } else {
      console.log("AnnotationStore: Already on the last page.");
      // Optionally, you could add logic to loop back to the first page if desired
    }
  }

  /**
   * Navigates to the previous page in the current manuscript.
   */
  function previousPage() {
    const pages = sortedPageIds.value;

    if (pages.length === 0) {
      console.log("AnnotationStore: No pages available to navigate.");
      return;
    }
    
    if (!currentPage.value) {
        // If current page is not set, it's ambiguous where "previous" should go.
        // Could go to last page, first page, or do nothing. Let's do nothing or go to first.
        console.log("AnnotationStore: Current page not set. Cannot determine previous page. Navigating to first page.");
        setCurrentPage(pages[0]);
        return;
    }

    const currentIndex = pages.indexOf(currentPage.value);
    if (currentIndex === -1) {
      console.warn(`AnnotationStore: Current page "${currentPage.value}" not found in available pages. Navigating to the first page.`);
      setCurrentPage(pages[0]); // Default to first page
      return;
    }

    if (currentIndex > 0) {
      setCurrentPage(pages[currentIndex - 1]);
    } else {
      console.log("AnnotationStore: Already on the first page.");
      // Optionally, you could add logic to loop back to the last page if desired
    }
  }

  /**
   * Sets an initial page, typically the first page of the manuscript.
   * Call this after `recognitions` data is loaded.
   */
  function setInitialPage() {
    const manuscript = currentManuscriptName.value;
    if (manuscript && recognitions.value[manuscript]) {
        const pages = sortedPageIds.value;
        if (pages.length > 0) {
            // Set to first page if current page is not already set or not in the list of pages
            if (!currentPage.value || !pages.includes(currentPage.value)) {
                 console.log(`AnnotationStore: Setting initial page to ${pages[0]}.`);
                 setCurrentPage(pages[0]);
            }
        } else {
            console.log("AnnotationStore: No pages available to set an initial page.");
            setCurrentPage(undefined); // Clear current page if no pages exist
        }
    } else {
        console.log("AnnotationStore: No manuscript data available to set an initial page.");
        setCurrentPage(undefined); // Clear current page if no manuscript
    }
  }

  // --- Existing functions (with minor robustness checks) ---

  function levenshteinDistance(str1 = '', str2 = '') {
    const track = Array(str2.length + 1)
      .fill(null)
      .map(() => Array(str1.length + 1).fill(null))
    for (let i = 0; i <= str1.length; i += 1) {
      track[0][i] = i
    }
    for (let j = 0; j <= str2.length; j += 1) {
      track[j][0] = j
    }
    for (let j = 1; j <= str2.length; j += 1) {
      for (let i = 1; i <= str1.length; i += 1) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1
        track[j][i] = Math.min(
          track[j][i - 1] + 1, // deletion
          track[j - 1][i] + 1, // insertion
          track[j - 1][i - 1] + indicator, // substitution
        )
      }
    }
    return track[str2.length][str1.length]
  }

  function calculateLevenshteinDistances() {
    for (const annotationsObject of userAnnotations.value) {
      const manuscript_name = annotationsObject['manuscript_name'];
      if (!recognitions.value[manuscript_name]) {
        console.warn(`Recognitions not found for manuscript: ${manuscript_name} during Levenshtein calculation.`);
        continue;
      }
      for (const page in annotationsObject['annotations']) {
        if (!recognitions.value[manuscript_name][page]) {
            console.warn(`Recognitions not found for page: ${page} in manuscript: ${manuscript_name} during Levenshtein calculation.`);
            continue;
        }
        for (const line in annotationsObject['annotations'][page]) {
          const recognitionLine = recognitions.value[manuscript_name][page][line];
          const annotationLine = annotationsObject['annotations'][page][line];

          if (recognitionLine && annotationLine &&
              typeof recognitionLine['predicted_label'] === 'string' &&
              typeof annotationLine['ground_truth'] === 'string'
            ) {
            annotationLine['levenshtein_distance'] =
              levenshteinDistance(
                recognitionLine['predicted_label'],
                annotationLine['ground_truth'],
              );
          } else {
            // console.warn(`Missing data for Levenshtein calculation: manuscript ${manuscript_name}, page ${page}, line ${line}`);
          }
        }
      }
    }
  }

  function exportToTxt() {
    const manuscript = currentManuscriptName.value;
    if (!manuscript || !recognitions.value[manuscript] || typeof recognitions.value[manuscript] !== 'object') {
      console.error("No valid manuscript data to export or manuscript name not found.");
      alert("No data available to export.");
      return;
    }

    const zipWriter = new zip.ZipWriter(new zip.BlobWriter("application/zip"));
    const pageKeys = Object.keys(recognitions.value[manuscript]);

    if (pageKeys.length === 0) {
        console.warn("No pages found in the manuscript to export.");
        alert("No pages found in the manuscript to export.");
        return;
    }

    pageKeys.forEach(pageName => {
      let lines = "";
      const pageContent = recognitions.value[manuscript][pageName]; // This is the page data { "0": {predicted_label: ""}, ... }
      
      // Assuming pageContent is an object where keys are line numbers/IDs (e.g. "0", "1")
      // and values are objects with a 'predicted_label'
      if (pageContent && typeof pageContent === 'object') {
        // Sort line keys numerically if possible, otherwise lexicographically
        const lineKeys = Object.keys(pageContent).sort((a, b) => {
            const numA = parseInt(a, 10); const numB = parseInt(b, 10);
            return (!isNaN(numA) && !isNaN(numB)) ? numA - numB : a.localeCompare(b);
        });

        lineKeys.forEach(lineKey => {
            const lineData = pageContent[lineKey];
            if (lineData && typeof lineData.predicted_label === 'string') {
              lines += lineData.predicted_label + "\n";
            } else {
              lines += "\n"; // Add an empty line if no label or incorrect format
            }
        });
      }
      zipWriter.add(`${pageName}.txt`, new zip.TextReader(lines));
    });

    zipWriter.close().then(blob => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${manuscript}_recognitions.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href); // Clean up blob URL
    }).catch(err => {
        console.error("Error creating zip file:", err);
        alert("Error creating zip file. Check console for details.");
    });
  }

  function reset() {
    modelName.value = null;
    recognitions.value = {};
    userAnnotations.value = [];
    currentPage.value = undefined; // Explicitly set to undefined
    console.log("AnnotationStore: Reset complete.");
  }

  return { 
    // State
    recognitions, 
    userAnnotations, 
    modelName, 
    currentPage, 
    
    // Computed (can be used by components if needed)
    currentManuscriptName, 
    sortedPageIds,

    // Actions
    setCurrentPage,
    nextPage,
    previousPage,
    setInitialPage, // Important for initializing after data load

    // Existing functions
    calculateLevenshteinDistances, 
    exportToTxt, 
    reset 
  };
});

// HMR (Hot Module Replacement)
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useAnnotationStore, import.meta.hot))
}