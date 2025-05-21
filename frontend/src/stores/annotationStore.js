import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref } from 'vue'
import * as zip from "@zip.js/zip.js";

export const useAnnotationStore = defineStore('annotations', () => {
  const modelName = ref()
  const recognitions = ref({})
  const userAnnotations = ref([])
  const currentPage = ref();

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
      const manuscript_name = annotationsObject['manuscript_name']
      for (const page in annotationsObject['annotations']) {
        for (const line in annotationsObject['annotations'][page]) {
          annotationsObject['annotations'][page][line]['levenshtein_distance'] =
            levenshteinDistance(
              recognitions.value[manuscript_name][page][line]['predicted_label'],
              annotationsObject['annotations'][page][line]['ground_truth'],
            )
        }
      }
    }
  }

  function exportToTxt() {
    const zipWriter = new zip.ZipWriter(new zip.BlobWriter("application/zip"));
    const manuscript_name = Object.keys(recognitions.value)[0];
    Object.keys(recognitions.value[manuscript_name]).forEach(pageName => {
      let lines = ""
      Object.keys(recognitions.value[manuscript_name][pageName]).forEach(lineNumber => {
        lines += (recognitions.value[manuscript_name][pageName][lineNumber]['predicted_label']) + "\n";
      })
      zipWriter.add(`${pageName}.txt`, new zip.TextReader(lines));
    })
    zipWriter.close().then(blob => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${manuscript_name}_recognitions.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

  }

  function reset() {
    modelName.value = null
    recognitions.value = {}
    userAnnotations.value = []
  }

  return { recognitions, userAnnotations, modelName, currentPage, calculateLevenshteinDistances, exportToTxt, reset }
})

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useAnnotationStore, import.meta.hot))
}
