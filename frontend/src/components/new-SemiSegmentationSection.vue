<template>
  <div class="manuscript-viewer">
    <!-- Top Toolbar: Collapsible -->
    <div class="toolbar">
      <h10>{{ manuscriptNameForDisplay }} - Page {{ currentPageForDisplay }}</h10>
      <div v-show="!isToolbarCollapsed" class="toolbar-controls">
        <button @click="previousPage" :disabled="loading || isProcessingSave || isFirstPage">
          Previous
        </button>
        <button @click="nextPage" :disabled="loading || isProcessingSave || isLastPage">Next</button>
        <button @click="saveAndGoNext" :disabled="loading || isProcessingSave">
          Save & Next (S)
        </button>
        <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">
          Annotate Text
        </button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="editModeActive" :disabled="isProcessingSave" />
            Edit Mode (W)
          </label>
        </div>
      </div>
      <button class="panel-toggle-btn" @click="isToolbarCollapsed = !isToolbarCollapsed">
        {{ isToolbarCollapsed ? 'Show Toolbar' : 'Hide' }}
      </button>
    </div>

    <!-- Main Content: Visualization Area -->
    <div class="visualization-container" ref="container">
      <div v-if="isProcessingSave" class="processing-save-notice">
        Saving graph and processing... Please wait.
      </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      <div v-if="loading" class="loading">Loading Page Data...</div>
      <div
        v-else
        class="image-container"
        :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
      >
        <img
          v-if="imageData"
          :src="`data:image/jpeg;base64,${imageData}`"
          :width="scaledWidth"
          :height="scaledHeight"
          class="manuscript-image"
          @load="imageLoaded = true"
        />
        <div
          v-else
          class="placeholder-image"
          :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
        >
          No image available
        </div>

        <svg
          v-if="graphIsLoaded"
          class="graph-overlay"
          :class="{ 'is-visible': editModeActive }"
          :width="scaledWidth"
          :height="scaledHeight"
          :style="{ cursor: svgCursor }"
          @click="editModeActive && onBackgroundClick($event)"
          @mousemove="handleSvgMouseMove"
          @mouseleave="handleSvgMouseLeave"
          ref="svgOverlayRef"
        >
          <line
            v-for="(edge, index) in workingGraph.edges"
            :key="`edge-${index}`"
            :x1="scaleX(workingGraph.nodes[edge.source].x)"
            :y1="scaleY(workingGraph.nodes[edge.source].y)"
            :x2="scaleX(workingGraph.nodes[edge.target].x)"
            :y2="scaleY(workingGraph.nodes[edge.target].y)"
            :stroke="getEdgeColor(edge)"
            :stroke-width="isEdgeSelected(edge) ? 3 : 2.5"
            @click.stop="editModeActive && onEdgeClick(edge, $event)"
          />

          <circle
            v-for="(node, nodeIndex) in workingGraph.nodes"
            :key="`node-${nodeIndex}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(nodeIndex)"
            :fill="getNodeColor(nodeIndex)"
            @click.stop="editModeActive && onNodeClick(nodeIndex, $event)"
          />

          <line
            v-if="
              editModeActive &&
              selectedNodes.length === 1 &&
              tempEndPoint &&
              !isAKeyPressed &&
              !isDKeyPressed
            "
            :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
            :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
            :x2="tempEndPoint.x"
            :y2="tempEndPoint.y"
            stroke="#ff9500"
            stroke-width="2.5"
            stroke-dasharray="5,5"
          />
        </svg>
      </div>
    </div>

    <!-- Bottom Panel: Collapsible -->
    <div class="bottom-panel">
      <div class="panel-toggle-bar" @click="isControlsCollapsed = !isControlsCollapsed">
        <div class="edit-instructions">
          <p v-if="isControlsCollapsed && editModeActive">
            Hold 'a' to connect, 'd' to delete. Press 's' to save & next. Toggle edit with 'w'.
          </p>
          <p v-else-if="isControlsCollapsed && !editModeActive">Press 'w' to enter edit mode.</p>
          <p v-else-if="!isAKeyPressed && !isDKeyPressed">
            Select nodes to manage edges, or use hotkeys.
          </p>
          <p v-else-if="isAKeyPressed">Release 'A' to connect nodes.</p>
          <p v-else-if="isDKeyPressed">Release 'D' to stop deleting.</p>
        </div>
        <button class="panel-toggle-btn">
          {{ isControlsCollapsed ? 'Show Controls' : 'Hide Controls' }}
        </button>
      </div>

      <div v-show="!isControlsCollapsed" class="bottom-panel-content">
        <div v-if="editModeActive && !isAKeyPressed && !isDKeyPressed" class="edit-controls">
          <div class="edit-actions">
            <button @click="resetSelection">Cancel Selection</button>
            <button
              @click="addEdge"
              :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Add Edge
            </button>
            <button
              @click="deleteEdge"
              :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Delete Edge
            </button>
          </div>
        </div>

        <div v-if="editModeActive && graphIsLoaded" class="modifications-log-container">
          <button @click="saveCurrentGraph" :disabled="loading || isProcessingSave">
            Save Graph
          </button>
          <div v-if="modifications.length > 0" class="modifications-details">
            <h3>Modifications ({{ modifications.length }})</h3>
            <button @click="resetModifications" :disabled="loading">Reset All Changes</button>
            <ul>
              <li
                v-for="(mod, index) in modifications"
                :key="index"
                class="modification-item"
              >
                {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge: {{ mod.source }} ↔
                {{ mod.target }}
                <button @click="undoModification(index)" class="undo-button">Undo</button>
              </li>
            </ul>
          </div>
          <p v-else-if="!loading">No modifications in this session.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js'
import { useRouter } from 'vue-router'

const props = defineProps({
  manuscriptName: {
    type: String,
    default: null,
  },
  pageName: {
    type: String,
    default: null,
  },
})

const router = useRouter()
const annotationStore = useAnnotationStore()

const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

const localManuscriptName = ref('')
const localCurrentPage = ref('')
const localPageList = ref([])

const loading = ref(true)
const isProcessingSave = ref(false)
const error = ref(null)
const imageData = ref('')
const imageLoaded = ref(false)

const isToolbarCollapsed = ref(true)
const isControlsCollapsed = ref(true)
const editModeActive = ref(true)

const dimensions = ref([0, 0])
const points = ref([])
const graph = ref({ nodes: [], edges: [] })
const workingGraph = reactive({ nodes: [], edges: [] })
const modifications = ref([])
const nodeEdgeCounts = ref({})
const selectedNodes = ref([])
const tempEndPoint = ref(null)
const isDKeyPressed = ref(false)
const isAKeyPressed = ref(false)
const hoveredNodesForMST = reactive(new Set())
const container = ref(null)
const svgOverlayRef = ref(null)

const scaleFactor = 1.0
const NODE_HOVER_RADIUS = 7
const EDGE_HOVER_THRESHOLD = 5

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(
  () => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1
)

const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor))
const scaleX = (x) => x * scaleFactor
const scaleY = (y) => y * scaleFactor
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)

const svgCursor = computed(() => {
  if (!editModeActive.value) return 'default'
  if (isAKeyPressed.value) return 'crosshair'
  if (isDKeyPressed.value) return 'not-allowed'
  return 'default'
})

const fetchPageData = async (manuscript, page) => {
  if (!manuscript || !page) {
    error.value = 'Manuscript or page not specified.'
    loading.value = false
    return
  }
  loading.value = true
  error.value = null
  modifications.value = []

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    imageData.value = data.image || ''
    points.value = data.points.map((p) => ({ coordinates: [p[0], p[1]], segment: null }))

    if (data.graph) {
      graph.value = data.graph
    } else if (data.points?.length > 0) {
      graph.value = generateLayoutGraph(data.points)
      // Only auto-save the initially generated graph in the "New Manuscript" flow
      if (!isEditModeFlow.value) {
        await saveGeneratedGraph(manuscript, page, graph.value)
      }
    }
    resetWorkingGraph()
  } catch (err) {
    console.error('Error fetching page data:', err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const fetchPageList = async (manuscript) => {
  if (!manuscript) return
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/manuscript/${manuscript}/pages`
    )
    if (!response.ok) throw new Error('Failed to fetch page list')
    localPageList.value = await response.json()
  } catch (err) {
    console.error('Failed to fetch page list:', err)
    localPageList.value = []
  }
}

const updateUniqueNodeEdgeCounts = () => {
  const counts = {}
  if (!workingGraph.nodes) return
  workingGraph.nodes.forEach((_, index) => {
    counts[index] = 0
  })

  if (!workingGraph.edges) {
    nodeEdgeCounts.value = counts
    return
  }

  const uniqueEdges = new Set()
  for (const edge of workingGraph.edges) {
    const key = `${Math.min(edge.source, edge.target)}-${Math.max(edge.source, edge.target)}`
    uniqueEdges.add(key)
  }

  for (const key of uniqueEdges) {
    const [source, target] = key.split('-').map(Number)
    if (counts[source] !== undefined) counts[source]++
    if (counts[target] !== undefined) counts[target]++
  }

  nodeEdgeCounts.value = counts
}

watch(() => workingGraph.edges, updateUniqueNodeEdgeCounts, { deep: true, immediate: true })

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []))
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []))
  resetSelection()
}

const getNodeColor = (nodeIndex) => {
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'
  if (isNodeSelected(nodeIndex)) return '#ff9500'

  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  if (edgeCount > 2) return '#2196F3'
  return '#cccccc'
}
const getNodeRadius = (nodeIndex) => {
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7
  if (isNodeSelected(nodeIndex)) return 6
  return edgeCount < 2 ? 5 : 3
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : '#ffffff')
const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex)
const isEdgeSelected = (edge) => {
  return (
    selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
      (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source))
  )
}

const resetSelection = () => {
  selectedNodes.value = []
  tempEndPoint.value = null
}
const onNodeClick = (nodeIndex, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value) return
  event.stopPropagation()
  const existingIndex = selectedNodes.value.indexOf(nodeIndex)
  if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1)
  else
    selectedNodes.value.length < 2
      ? selectedNodes.value.push(nodeIndex)
      : (selectedNodes.value = [nodeIndex])
}
const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value) return
  event.stopPropagation()
  selectedNodes.value = [edge.source, edge.target]
}
const onBackgroundClick = () => {
  if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection()
}
const handleSvgMouseMove = (event) => {
  if (!editModeActive.value || !svgOverlayRef.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  if (isDKeyPressed.value) handleEdgeHoverDelete(mouseX, mouseY)
  else if (isAKeyPressed.value) handleNodeHoverCollect(mouseX, mouseY)
  else if (selectedNodes.value.length === 1) tempEndPoint.value = { x: mouseX, y: mouseY }
  else tempEndPoint.value = null
}
const handleSvgMouseLeave = () => {
  if (selectedNodes.value.length === 1) tempEndPoint.value = null
}
const handleGlobalKeyDown = (e) => {
  if (e.key.toLowerCase() === 'w' && !e.repeat) {
    e.preventDefault()
    editModeActive.value = !editModeActive.value
    return
  }

  if (!editModeActive.value || e.repeat) return

  const key = e.key.toLowerCase()
  if (key === 's') {
    e.preventDefault()
    if (!loading.value && !isProcessingSave.value) saveAndGoNext()
  }
  if (key === 'd') {
    e.preventDefault()
    isDKeyPressed.value = true
    resetSelection()
  }
  if (key === 'a') {
    e.preventDefault()
    isAKeyPressed.value = true
    hoveredNodesForMST.clear()
    resetSelection()
  }
}
const handleGlobalKeyUp = (e) => {
  if (!editModeActive.value) return

  const key = e.key.toLowerCase()
  if (key === 'd') isDKeyPressed.value = false
  if (key === 'a') {
    isAKeyPressed.value = false
    if (hoveredNodesForMST.size >= 2) addMSTEdges()
    hoveredNodesForMST.clear()
  }
}

const edgeExists = (nodeA, nodeB) =>
  workingGraph.edges.some(
    (e) =>
      (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA)
  )
const addEdge = () => {
  if (selectedNodes.value.length !== 2 || edgeExists(...selectedNodes.value)) return
  const [source, target] = selectedNodes.value
  const newEdge = { source, target, label: 0, modified: true }
  workingGraph.edges.push(newEdge)
  modifications.value.push({ type: 'add', source, target, label: 0 })
  resetSelection()
}
const deleteEdge = () => {
  if (selectedNodes.value.length !== 2) return
  const [source, target] = selectedNodes.value
  const edgeIndex = workingGraph.edges.findIndex(
    (e) =>
      (e.source === source && e.target === target) || (e.source === target && e.target === source)
  )
  if (edgeIndex === -1) return
  const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0]
  modifications.value.push({
    type: 'delete',
    source: removedEdge.source,
    target: removedEdge.target,
    label: removedEdge.label,
  })
  resetSelection()
}
const undoModification = (index) => {
  const mod = modifications.value.splice(index, 1)[0]
  if (mod.type === 'add') {
    const edgeIndex = workingGraph.edges.findIndex(
      (e) => e.source === mod.source && e.target === mod.target
    )
    if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1)
  } else if (mod.type === 'delete') {
    workingGraph.edges.push({
      source: mod.source,
      target: mod.target,
      label: mod.label,
      modified: true,
    })
  }
}
const resetModifications = () => {
  resetWorkingGraph()
  modifications.value = []
}

const distanceToLineSegment = (px, py, x1, y1, x2, y2) =>
  Math.hypot(
    px -
      (x1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (x2 - x1)),
    py -
      (y1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (y2 - y1))
  )
const handleEdgeHoverDelete = (mouseX, mouseY) => {
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i]
    const n1 = workingGraph.nodes[edge.source],
      n2 = workingGraph.nodes[edge.target]
    if (
      n1 &&
      n2 &&
      distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) <
        EDGE_HOVER_THRESHOLD
    ) {
      const removed = workingGraph.edges.splice(i, 1)[0]
      modifications.value.push({
        type: 'delete',
        source: removed.source,
        target: removed.target,
        label: removed.label,
      })
    }
  }
}
const handleNodeHoverCollect = (mouseX, mouseY) => {
  workingGraph.nodes.forEach((node, index) => {
    if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS)
      hoveredNodesForMST.add(index)
  })
}
const calculateMST = (indices, nodes) => {
  const points = indices.map((i) => ({ ...nodes[i], originalIndex: i }))
  const edges = []
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      edges.push({
        source: points[i].originalIndex,
        target: points[j].originalIndex,
        weight: Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y),
      })
    }
  edges.sort((a, b) => a.weight - b.weight)
  const parent = {}
  indices.forEach((i) => (parent[i] = i))
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  const union = (i, j) => {
    const rootI = find(i),
      rootJ = find(j)
    if (rootI !== rootJ) {
      parent[rootJ] = rootI
      return true
    }
    return false
  }
  return edges.filter((e) => union(e.source, e.target))
}
const addMSTEdges = () => {
  calculateMST(Array.from(hoveredNodesForMST), workingGraph.nodes).forEach((edge) => {
    if (!edgeExists(edge.source, edge.target)) {
      const newEdge = { source: edge.source, target: edge.target, label: 0, modified: true }
      workingGraph.edges.push(newEdge)
      modifications.value.push({ type: 'add', ...newEdge })
    }
  })
}

const saveGeneratedGraph = async (name, page, g) => {
  try {
    await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-graph/${name}/${page}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: g }),
    })
  } catch (e) {
    console.error('Error saving generated graph:', e)
  }
}

const saveModifications = async () => {
  // Construct the base request body
  const requestBody = {
    graph: workingGraph,
    modifications: modifications.value,
  }

  // Conditionally add modelName only if it exists.
  // This is the key fix to prevent sending 'default.pth'.
  if (annotationStore.modelName) {
    requestBody.modelName = annotationStore.modelName
  }

  try {
    const res = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${localManuscriptName.value}/${localCurrentPage.value}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }
    )
    if (!res.ok) throw new Error((await res.json()).error || 'Save failed')
    const data = await res.json()
    graph.value = JSON.parse(JSON.stringify(workingGraph))
    modifications.value = []
    // If we are in the new manuscript flow and recognition was run, update the store.
    if (!isEditModeFlow.value && data.lines) {
      annotationStore.recognitions[localManuscriptName.value][localCurrentPage.value] = data.lines
    }
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err // Re-throw to be caught by calling functions
  }
}

const saveCurrentGraph = async () => {
  if (isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    alert('Graph saved!')
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value) return
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before navigating?')) {
      isProcessingSave.value = true
      try {
        await saveModifications()
        navAction()
      } catch (err) {
        alert('Save failed, navigation cancelled.')
      } finally {
        isProcessingSave.value = false
      }
    } else {
      modifications.value = [] // Discard changes
      navAction()
    }
  } else {
    navAction()
  }
}

const navigateToPage = (page) => {
  if (isEditModeFlow.value) {
    router.push({
      name: 'edit-manuscript-layout',
      params: { manuscriptName: localManuscriptName.value, pageName: page },
    })
  } else {
    annotationStore.setCurrentPage(page)
  }
}

const previousPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex > 0) {
      navigateToPage(localPageList.value[currentIndex - 1])
    }
  })

const nextPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    }
  })
const goToIMG2TXTPage = () => {
  if (isEditModeFlow.value) {
    alert(
      "Text annotation is part of the 'New Manuscript' flow. This action is disabled in edit mode."
    )
    return
  }
  confirmAndNavigate(() => router.push({ name: 'img-2-txt' }))
}

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    } else {
      alert('Last page saved successfully!')
    }
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

onMounted(async () => {
  if (isEditModeFlow.value) {
    localManuscriptName.value = props.manuscriptName
    localCurrentPage.value = props.pageName
    await fetchPageList(props.manuscriptName)
    await fetchPageData(props.manuscriptName, props.pageName)
  } else {
    localManuscriptName.value = Object.keys(annotationStore.recognitions)[0] || ''
    localPageList.value = annotationStore.sortedPageIds
    localCurrentPage.value = annotationStore.currentPage
    if (localManuscriptName.value && localCurrentPage.value) {
      await fetchPageData(localManuscriptName.value, localCurrentPage.value)
    } else {
      loading.value = false
      error.value = "No manuscript data found. Please start from the 'New Manuscript' page."
    }
  }

  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
})

watch(
  () => annotationStore.currentPage,
  (newPage) => {
    if (!isEditModeFlow.value && newPage && newPage !== localCurrentPage.value) {
      localCurrentPage.value = newPage
      fetchPageData(localManuscriptName.value, newPage)
    }
  }
)

watch(
  () => props.pageName,
  (newPageName) => {
    if (isEditModeFlow.value && newPageName && newPageName !== localCurrentPage.value) {
      localCurrentPage.value = newPageName
      fetchPageData(localManuscriptName.value, newPageName)
    }
  }
)

watch(editModeActive, (isEditing) => {
  if (!isEditing) {
    resetSelection()
    isAKeyPressed.value = false
    isDKeyPressed.value = false
    hoveredNodesForMST.clear()
  }
})
</script>
<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  background-color: #333;
  color: #fff;
}

/* --- Toolbar --- */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background-color: #424242;
  border-bottom: 1px solid #555;
  flex-shrink: 0;
  gap: 16px;
}
.toolbar-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

/* --- Main Visualization Area --- */
.visualization-container {
  position: relative;
  overflow: auto;
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 1rem;
}
.image-container {
  position: relative;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
}
.manuscript-image {
  display: block;
  user-select: none;
  opacity: 0.7;
}
.graph-overlay {
  position: absolute;
  top: 0;
  left: 0;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease-in-out;
}
.graph-overlay.is-visible {
  opacity: 1;
  pointer-events: auto;
}

/* --- Bottom Panel --- */
.bottom-panel {
  background-color: #4f4f4f;
  border-top: 1px solid #555;
  flex-shrink: 0;
  transition: all 0.3s ease;
}
.panel-toggle-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  cursor: pointer;
}
.edit-instructions p {
  margin: 0;
  font-size: 0.9em;
  color: #ccc;
  font-style: italic;
}
.bottom-panel-content {
  padding: 10px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.edit-controls,
.modifications-log-container {
  display: flex;
  align-items: flex-start;
  gap: 20px;
}
.edit-actions {
  display: flex;
  gap: 8px;
}

/* --- UI Elements & States --- */
.panel-toggle-btn {
  padding: 4px 10px;
  font-size: 0.8em;
  background-color: #616161;
  border: 1px solid #757575;
}
.processing-save-notice,
.loading,
.error-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 20px 30px;
  border-radius: 8px;
  z-index: 10000;
  text-align: center;
}
.processing-save-notice {
  background-color: rgba(0, 0, 0, 0.8);
}
.error-message {
  background-color: #c62828;
}
.loading {
  font-size: 1.2rem;
  color: #aaa;
  background: none;
}
button {
  padding: 6px 14px;
  border-radius: 4px;
  border: 1px solid #666;
  background-color: #555;
  color: #fff;
  cursor: pointer;
  transition: background-color 0.2s ease;
}
button:hover:not(:disabled) {
  background-color: #6a6a6a;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* --- Modifications Log --- */
.modifications-details {
  flex-grow: 1;
}
.modifications-details h3 {
  margin: 0 0 8px 0;
  font-size: 1.1em;
  color: #eee;
}
.modifications-details ul {
  list-style-type: none;
  padding: 0;
  max-height: 120px;
  overflow-y: auto;
  border: 1px solid #666;
  background-color: #3e3e3e;
  border-radius: 3px;
}
.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  border-bottom: 1px solid #555;
  font-size: 0.9em;
}
.modification-item:last-child {
  border-bottom: none;
}
.undo-button {
  background-color: #6d6d3d;
  border-color: #888855;
}
.undo-button:hover:not(:disabled) {
  background-color: #7a7a4a;
}
</style>