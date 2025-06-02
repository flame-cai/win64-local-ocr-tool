<template>
  <div class="manuscript-viewer">
    <div class="toolbar">
      <h9>{{ manuscriptName }} - Page {{ currentPage }}</h9>
      <div class="controls">
        <button @click="previousPage" :disabled="loading || isProcessingSave">Previous</button>
        <button @click="nextPage" :disabled="loading || isProcessingSave">Next</button>
        <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">Annotate Text</button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="editMode" :disabled="isProcessingSave" />
            Edit Mode
          </label>
        </div>
        <br>
        <div class="edit-instructions">
          <p v-if="selectedNodes.length === 0">Hold 'a' and hover over nodes to connect them.<br>Hold 'd' and hover over edges to delete them.</p>
          <p v-else-if="selectedNodes.length === 1">Click a second node to select it for creating/deleting an edge, or click background/another node to change selection.</p>
          <p v-else>Click "Add Edge" or "Delete Edge" below, or click background/another node to change selection.</p>
        </div>
      </div>
    </div>
    <div v-if="isProcessingSave" class="processing-save-notice">
      Saving graph and processing... Please wait.
    </div>


    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="loading" class="loading">
    </div>

    <div v-else class="visualization-container" ref="container">
      <div class="image-container" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
        <img 
          v-if="imageData" 
          :src="`data:image/jpeg;base64,${imageData}`" 
          :width="scaledWidth" 
          :height="scaledHeight" 
          class="manuscript-image"
          @load="imageLoaded = true"
        />
        <div v-else class="placeholder-image" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
          No image available
        </div>
        
        <div 
          v-if="effectiveShowPoints && points.length > 0" 
          class="points-overlay"
        >
          <div 
            v-for="(point, index) in points" 
            :key="`point-${index}`"
            class="point"
            :style="{
              left: `${scaleX(point.coordinates[0])}px`,
              top: `${scaleY(point.coordinates[1])}px`
            }"
            :title="`Point ${index}: (${point.coordinates[0]}, ${point.coordinates[1]})`"
          ></div>
        </div>
        
        <svg 
          v-if="effectiveShowGraph && workingGraph.nodes && workingGraph.nodes.length > 0" 
          class="graph-overlay"
          :width="scaledWidth"
          :height="scaledHeight"
          @click="editMode && onBackgroundClick"
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
            :stroke-opacity="1"
            @click.stop="editMode && onEdgeClick(edge, $event)"
          />
          
          <circle
            v-for="(node, nodeIndex) in workingGraph.nodes"
            :key="`node-${nodeIndex}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(nodeIndex)"
            :fill="getNodeColor(nodeIndex)"
            :fill-opacity="1"
            @click.stop="editMode && onNodeClick(nodeIndex, $event)"
          />
          
          <line
            v-if="editMode && selectedNodes.length === 1 && tempEndPoint && !isAKeyPressed && !isDKeyPressed"
            :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
            :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
            :x2="tempEndPoint.x"
            :y2="tempEndPoint.y"
            stroke="#ff9500"
            stroke-width="2.5"
            stroke-dasharray="5,5"
            stroke-opacity="1"
          />
        </svg>
      </div>
    </div>

    <div v-if="editMode && !isAKeyPressed && !isDKeyPressed" class="edit-controls">
      <div class="edit-actions">
        <button @click="resetSelection">Cancel Selection</button>
        <button 
          @click="addEdge" 
          :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
        >Add Edge</button>
        <button 
          @click="deleteEdge" 
          :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
        >Delete Edge</button>
      </div>
    </div>
     <div v-else-if="editMode && isAKeyPressed" class="edit-controls">
      <p>Release 'A' to connect hovered nodes with a Minimum Spanning Tree.</p>
    </div>
    <div v-else-if="editMode && isDKeyPressed" class="edit-controls">
      <p>Hover over edges to delete them. Release 'D' to stop.</p>
    </div>


    <div v-if="editMode && graphIsLoaded" class="modifications-log-container">
        <button @click="saveModifications" :disabled="loading">Save Graph</button>
        <div v-if="modifications.length > 0" class="modifications-details">
            <h3>Modifications ({{ modifications.length }})</h3>
            <button @click="resetModifications" :disabled="loading">Reset All Changes</button>
            <ul>
              <li v-for="(mod, index) in modifications" :key="index" class="modification-item">
                {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge between Node {{ mod.source }} and Node {{ mod.target }}
                <button @click="undoModification(index)" class="undo-button">Undo</button>
              </li>
            </ul>
        </div>
        <p v-else-if="!loading && workingGraph.nodes && workingGraph.nodes.length > 0">No modifications made in this session.</p>
    </div>

  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue';
import { useAnnotationStore } from '@/stores/annotationStore';
import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js';
import { useRouter } from 'vue-router';

const router = useRouter();
const annotationStore = useAnnotationStore();

const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
const currentPage = computed(() => annotationStore.currentPage);
const isProcessingSave = ref(false); // For UX during save before navigation
const loading = ref(true);
const error = ref(null);
const dimensions = ref([0, 0]);
const points = ref([]);
const graph = ref({ nodes: [], edges: [] });
const imageData = ref('');
const imageLoaded = ref(false);

const editMode = ref(true); // Edit mode is ON by default
const selectedNodes = ref([]);
const tempEndPoint = ref(null);
const modifications = ref([]);
const workingGraph = reactive({ nodes: [], edges: [] });

const scaleFactor = 0.5;
const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor));
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor));
const scaleX = (x) => x * scaleFactor;
const scaleY = (y) => y * scaleFactor;

const container = ref(null);
const svgOverlayRef = ref(null); // Ref for the SVG element

// New state for hover interactions
const isDKeyPressed = ref(false);
const isAKeyPressed = ref(false);
const hoveredNodesForMST = reactive(new Set());
const NODE_HOVER_RADIUS = 2; // Pixels on scaled view for node proximity
const EDGE_HOVER_THRESHOLD = 2; // Pixels on scaled view for edge proximity

// Computed properties for UI elements linked to editMode
const effectiveShowPoints = computed(() => editMode.value);
const effectiveShowGraph = computed(() => editMode.value);
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0);

const goToIMG2TXTPage = async () => {
  if (isProcessingSave.value) return; // Prevent double-clicks

  if (editMode.value && graphIsLoaded.value) {
    isProcessingSave.value = true;
    try {
      console.log("Attempting to save graph before navigating to Annotate Text...");
      await saveModifications(); // saveModifications should handle its own error display if needed
      console.log("Graph saved successfully. Navigating to Annotate Text.");
      router.push({ name: 'img-2-txt' });
    } catch (err) {
      // saveModifications should ideally set its own error ref
      // or this component can display a generic error based on the catch
      console.error("Failed to save graph before navigating:", err);
      // Optionally, display an alert or a more prominent error message to the user
      alert(`Error saving graph: ${err.message}. Cannot proceed to Annotate Text.`);
    } finally {
      isProcessingSave.value = false;
    }
  } else {
    // If not in edit mode or no graph is loaded, just navigate
    console.log("Not in edit mode or no graph loaded. Navigating directly to Annotate Text.");
    router.push({ name: 'img-2-txt' });
  }
};

const updateCanvasSize = (width, height) => {
  dimensions.value = [width, height];
};

const saveGeneratedGraph = async (manuscriptName, page, graphData) => {
  try {
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + `/save-graph/${manuscriptName}/${page}`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ graph: graphData }) }
    );
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to save graph');
    return await response.json();
  } catch (error) {
    console.error('Error saving graph to backend:', error);
    return null;
  }
};

const fetchPageData = async () => {
  if (!manuscriptName.value || !currentPage.value) return;
  loading.value = true;
  error.value = null;
  points.value = [];
  graph.value = { nodes: [], edges: [] };
  imageData.value = '';
  imageLoaded.value = false;
  modifications.value = []; // Clear modifications for new page
  
  try {
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + `/semi-segment/${manuscriptName.value}/${currentPage.value}`
    );
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data');
    const data = await response.json();
    
    updateCanvasSize(data.dimensions[0], data.dimensions[1]);
    points.value = data.points.map(point => ({ coordinates: [point[0], point[1]], segment: null }));
    
    if (data.graph) {
      graph.value = data.graph;
    } else if (data.points && data.points.length > 0) {
      const generatedGraph = generateLayoutGraph(data.points);
      graph.value = generatedGraph;
      await saveGeneratedGraph(manuscriptName.value, currentPage.value, generatedGraph);
    }
    resetWorkingGraph();
    if (data.image) imageData.value = data.image;

  } catch (err) {
    console.error('Error fetching page data:', err);
    error.value = err.message || 'Failed to load page data';
  } finally {
    loading.value = false;
  }
};

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
  resetSelection();
  // modifications.value = []; // This is handled per page load or explicitly by user
};

const resetSelection = () => {
  selectedNodes.value = [];
  tempEndPoint.value = null;
};

const onNodeClick = (nodeIndex, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value) return; // Prevent selection during hover actions
  event.stopPropagation();
  const existingIndex = selectedNodes.value.indexOf(nodeIndex);
  if (existingIndex !== -1) {
    selectedNodes.value.splice(existingIndex, 1);
  } else {
    if (selectedNodes.value.length < 2) {
      selectedNodes.value.push(nodeIndex);
    } else {
      selectedNodes.value = [nodeIndex];
    }
  }
  tempEndPoint.value = null;
};

const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value) return; // Prevent selection during hover actions
  event.stopPropagation();
  selectedNodes.value = [edge.source, edge.target];
};

const onBackgroundClick = () => {
  if (isAKeyPressed.value || isDKeyPressed.value) return;
  resetSelection();
};

const edgeExists = (nodeA, nodeB) => {
  return workingGraph.edges.some(e => 
    (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA)
  );
};

const addEdgeManual = () => { // Renamed to avoid conflict with potential future generic addEdge
  if (selectedNodes.value.length !== 2) return;
  const [source, target] = selectedNodes.value;
  if (edgeExists(source, target)) return;
  
  const newEdge = { source, target, label: 0, modified: true };
  workingGraph.edges.push(newEdge);
  modifications.value.push({ type: 'add', ...newEdge });
  resetSelection();
};
// Keep original name for button call compatibility
const addEdge = addEdgeManual;


const deleteEdgeManual = () => { // Renamed to avoid conflict
  if (selectedNodes.value.length !== 2) return;
  const [source, target] = selectedNodes.value;
  const edgeIndex = workingGraph.edges.findIndex(e => 
    (e.source === source && e.target === target) || (e.source === target && e.target === source)
  );
  if (edgeIndex === -1) return;
  
  const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0];
  modifications.value.push({ type: 'delete', source: removedEdge.source, target: removedEdge.target, label: removedEdge.label });
  resetSelection();
};
// Keep original name for button call compatibility
const deleteEdge = deleteEdgeManual;


const undoModification = (index) => {
  const mod = modifications.value[index];
  if (mod.type === 'add') {
    const edgeIndex = workingGraph.edges.findIndex(e => 
      (e.source === mod.source && e.target === mod.target) || (e.source === mod.target && e.target === mod.source)
    );
    if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1);
  } else if (mod.type === 'delete') {
    workingGraph.edges.push({ source: mod.source, target: mod.target, label: mod.label, modified: true }); // Re-added edge is modified from original state
  }
  modifications.value.splice(index, 1);
};

const resetModifications = () => {
  resetWorkingGraph(); // This reloads from original graph.value
  modifications.value = []; // And clears modification log
};

const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex);

const isEdgeSelected = (edge) => {
  return selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
     (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));
};

const getEdgeColor = (edge) => {
  if (edge.modified) return '#f44336'; // Highlight modified edges
  return edge.label === 0 ? '#ffffff' : '#e74c3c'; // Original logic
};

const getNodeColor = (nodeIndex) => {
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'; // Cyan for hover-collect
  return isNodeSelected(nodeIndex) ? '#ff9500' : '#f44336';
};

const getNodeRadius = (nodeIndex) => {
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 5;
  return isNodeSelected(nodeIndex) ? 6 : 3;
};

// Navigation with save confirmation
const confirmAndNavigate = async (navigationAction) => {
  if (isProcessingSave.value) {
    alert("Please wait for the current save operation to complete.");
    return;
  }
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before navigating?')) {
      isProcessingSave.value = true;
      try {
        await saveModifications();
        navigationAction();
      } catch (err) {
        console.error("Failed to save, navigation cancelled:", err);
        alert("Failed to save changes. Please try again or discard changes to navigate.");
      } finally {
        isProcessingSave.value = false;
      }
    } else {
      modifications.value = []; // Discard changes
      navigationAction();
    }
  } else {
    navigationAction();
  }
};

const nextPage = () => confirmAndNavigate(() => annotationStore.nextPage());
const previousPage = () => confirmAndNavigate(() => annotationStore.previousPage());

// --- New Hover Interaction Logic ---

const handleGlobalKeyDown = (e) => {
  if (!editMode.value || e.repeat) return; // Only in edit mode, ignore repeats for initial press

  if (e.key.toLowerCase() === 'd') {
    e.preventDefault();
    isDKeyPressed.value = true;
    resetSelection(); // Clear selection when starting hover-delete
  }
  if (e.key.toLowerCase() === 'a') {
    e.preventDefault();
    isAKeyPressed.value = true;
    hoveredNodesForMST.clear();
    resetSelection(); // Clear selection when starting hover-add
  }
};

const handleGlobalKeyUp = (e) => {
  if (!editMode.value) return;

  if (e.key.toLowerCase() === 'd') {
    isDKeyPressed.value = false;
  }
  if (e.key.toLowerCase() === 'a') {
    isAKeyPressed.value = false;
    if (hoveredNodesForMST.size >= 2) {
      addMSTEdges();
    }
    hoveredNodesForMST.clear();
  }
};

const handleSvgMouseMove = (event) => {
  if (!editMode.value || !svgOverlayRef.value) return;

  const svgRect = svgOverlayRef.value.getBoundingClientRect();
  const mouseX = event.clientX - svgRect.left;
  const mouseY = event.clientY - svgRect.top;

  if (isDKeyPressed.value) {
    handleEdgeHoverDelete(mouseX, mouseY);
  } else if (isAKeyPressed.value) {
    handleNodeHoverCollect(mouseX, mouseY);
  } else if (selectedNodes.value.length === 1) {
    tempEndPoint.value = { x: mouseX, y: mouseY };
  } else {
    tempEndPoint.value = null;
  }
};

const handleSvgMouseLeave = () => {
    // If 'a' is not held, clear tempEndPoint when mouse leaves SVG
    if (selectedNodes.value.length === 1 && !isAKeyPressed.value && !isDKeyPressed.value) {
        tempEndPoint.value = null;
    }
    // Note: hoveredNodesForMST is cleared on 'a' keyup, not on mouse leave.
};


function distanceToLineSegment(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
  if (l2 === 0) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  const projX = x1 + t * (x2 - x1);
  const projY = y1 + t * (y2 - y1);
  return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

const handleEdgeHoverDelete = (mouseX, mouseY) => {
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i];
    const nodeSource = workingGraph.nodes[edge.source];
    const nodeTarget = workingGraph.nodes[edge.target];

    if (!nodeSource || !nodeTarget) continue;

    const x1 = scaleX(nodeSource.x);
    const y1 = scaleY(nodeSource.y);
    const x2 = scaleX(nodeTarget.x);
    const y2 = scaleY(nodeTarget.y);

    const dist = distanceToLineSegment(mouseX, mouseY, x1, y1, x2, y2);

    if (dist < EDGE_HOVER_THRESHOLD) {
      const removedEdge = workingGraph.edges.splice(i, 1)[0];
      modifications.value.push({
        type: 'delete',
        source: removedEdge.source,
        target: removedEdge.target,
        label: removedEdge.label,
      });
      // No break, continue checking other edges to delete all under cursor (sweep delete)
    }
  }
};

const handleNodeHoverCollect = (mouseX, mouseY) => {
  workingGraph.nodes.forEach((node, index) => {
    const nodeX = scaleX(node.x);
    const nodeY = scaleY(node.y);
    const distSq = (mouseX - nodeX) ** 2 + (mouseY - nodeY) ** 2;
    if (distSq < NODE_HOVER_RADIUS ** 2) {
      hoveredNodesForMST.add(index);
    }
  });
};

class DSU {
  constructor() {
    this.parent = [];
    this.nodeMap = new Map(); 
    this.reverseNodeMap = new Map(); 
  }

  init(nodeIndices) {
    this.parent = Array(nodeIndices.length).fill(0).map((_, i) => i);
    this.nodeMap.clear();
    this.reverseNodeMap.clear();
    nodeIndices.forEach((originalIndex, dsuIndex) => {
      this.nodeMap.set(originalIndex, dsuIndex);
      this.reverseNodeMap.set(dsuIndex, originalIndex);
    });
  }
  
  find(originalNodeIndex) {
    const i = this.nodeMap.get(originalNodeIndex);
    if (this.parent[i] === i) return i;
    // Path compression: map result of find back to original node index space, then find again
    const rootOriginalNodeIndex = this.reverseNodeMap.get(this.parent[i]);
    const rootDsuIndex = this.find(rootOriginalNodeIndex); // find expects original index
    this.parent[i] = rootDsuIndex; // Store DSU index
    return rootDsuIndex;
  }

  union(originalNodeIndex1, originalNodeIndex2) {
    const root1DsuIndex = this.find(originalNodeIndex1);
    const root2DsuIndex = this.find(originalNodeIndex2);
    if (root1DsuIndex !== root2DsuIndex) {
      this.parent[root2DsuIndex] = root1DsuIndex;
      return true;
    }
    return false;
  }
}

function calculateMST(nodeIndices, allNodesData) {
  if (nodeIndices.length < 2) return [];

  const pointsData = nodeIndices.map(index => ({ ...allNodesData[index], originalIndex: index }));
  const mstEdges = [];

  const potentialEdges = [];
  for (let i = 0; i < pointsData.length; i++) {
    for (let j = i + 1; j < pointsData.length; j++) {
      const p1 = pointsData[i];
      const p2 = pointsData[j];
      const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
      potentialEdges.push({ source: p1.originalIndex, target: p2.originalIndex, weight: dist });
    }
  }

  potentialEdges.sort((a, b) => a.weight - b.weight);

  const dsu = new DSU();
  dsu.init(nodeIndices);

  for (const edge of potentialEdges) {
    if (dsu.union(edge.source, edge.target)) {
      mstEdges.push({ source: edge.source, target: edge.target });
    }
  }
  return mstEdges;
}

const addMSTEdges = () => {
  const nodesToConnect = Array.from(hoveredNodesForMST);
  if (nodesToConnect.length < 2) return;

  const mstNewEdges = calculateMST(nodesToConnect, workingGraph.nodes);
  mstNewEdges.forEach(edge => {
    if (!edgeExists(edge.source, edge.target)) {
      const newEdge = {
        source: edge.source,
        target: edge.target,
        label: 0,
        modified: true,
      };
      workingGraph.edges.push(newEdge);
      modifications.value.push({ type: 'add', ...newEdge });
    }
  });
};

// Watch for page changes
watch(() => annotationStore.currentPage, (newPage, oldPage) => {
    if (isProcessingSave.value) {
      console.warn("Page change triggered while processing save, deferring fetchPageData.");
      // Decide if you want to queue the page change or prevent it
      return;
    }
    if (newPage && newPage !== oldPage) {
      fetchPageData();
    } else if (!newPage && oldPage) { 
      points.value = [];
      graph.value = { nodes: [], edges: [] };
      // ... reset other page specific data
      modifications.value = [];
      resetWorkingGraph();
      loading.value = false;
      error.value = null;
    }
  },
  { immediate: true }
);

// Watch for edit mode toggle
watch(editMode, (newValue) => {
  if (!newValue) { // Exiting edit mode
    resetSelection();
    isAKeyPressed.value = false;
    isDKeyPressed.value = false;
    hoveredNodesForMST.clear();
    tempEndPoint.value = null;
  }
});

onMounted(() => {
  window.addEventListener('keydown', handleGlobalKeyDown);
  window.addEventListener('keyup', handleGlobalKeyUp);
  // Initial fetch if currentPage is already set (e.g. on page refresh)
  if (annotationStore.currentPage && !imageLoaded.value && !loading.value && !isProcessingSave.value) {
      fetchPageData();
  }
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown);
  window.removeEventListener('keyup', handleGlobalKeyUp);
});

const saveModifications = async () => {
  // Ensure isProcessingSave is also managed if this function is called directly
  // However, it's better to have a separate function for the button if behavior needs to differ
  // For now, assuming saveModifications is the core save logic.
  // loading.value = true; // This is handled by the saveModificationsAndStay or by goToIMG2TXTPage's isProcessingSave

  // The original saveModifications logic:
  try {
    console.log('Saving modifications and generating line labels...');
    
    const request = {
          graph: workingGraph,
          modifications: modifications.value,
          points: points.value.map(point => point.segment),
          modelName: annotationStore.modelName
        };
    
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Failed to parse error response from backend" }));
      console.error('Backend error response during save/recognition:', errorData);
      throw new Error(errorData.error || 'Failed to save modifications and generate labels');
    }

    const responseData = await response.json();
    console.log('Save and recognition RESPONSE DATA:', responseData);

    // Update the original graph state in this component
    graph.value = JSON.parse(JSON.stringify(workingGraph)); // graph is the 'base' graph
    modifications.value = []; // Clear modifications log as they are now saved

    if (responseData.lines) {
      if (!annotationStore.recognitions[manuscriptName.value]) {
        annotationStore.recognitions[manuscriptName.value] = {};
      }
      annotationStore.recognitions[manuscriptName.value][currentPage.value] = responseData.lines;
      console.log(`Line data updated in store for manuscript '${manuscriptName.value}', page '${currentPage.value}'.`);
    } else {
      console.warn('NO responseData.lines received in response from /semi-segment POST.');
      if (annotationStore.recognitions[manuscriptName.value] && !annotationStore.recognitions[manuscriptName.value][currentPage.value]) {
         console.warn('Initializing empty page data in store because responseData.lines was missing.');
         annotationStore.recognitions[manuscriptName.value][currentPage.value] = {};
      }
    }
    error.value = null; // Clear previous errors
    console.log('Graph modifications saved and page recognized successfully.');
  } catch (err) {
    console.error('Error saving modifications:', err);
    error.value = err.message || 'Failed to save modifications';
    throw err; // Re-throw the error so the calling function (goToIMG2TXTPage) knows it failed
  } finally {
    // loading.value = false; // Handled by the wrapper
  }
};

// New function for the "Save Graph" button to provide its own loading/processing state if needed
const saveModificationsAndStay = async () => {
  if (isProcessingSave.value) return; // Prevent double processing

  isProcessingSave.value = true;
  try {
    await saveModifications();
    // Optionally, show a success message like "Graph saved!"
    alert("Graph saved successfully!"); // Simple feedback
  } catch (err) {
    // Error is already logged by saveModifications
    // Optionally, show an error message like "Failed to save graph."
    alert(`Failed to save graph: ${err.message}`); // Simple feedback
  } finally {
    isProcessingSave.value = false;
  }
};

</script>

<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  overflow: hidden; /* Prevent tool from overflowing viewport */
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0; /* Toolbar should not shrink */
}

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.toolbar button:disabled,
.modifications-log-container button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.toggle-container label input[type="checkbox"]:disabled + span { /* If you wrap text in span */
  opacity: 0.6;
  cursor: not-allowed;
}
.toggle-container label input[type="checkbox"]:disabled {
  cursor: not-allowed;
}


.toggle-container {
  display: flex;
  gap: 8px;
}
.toggle-container label {
  display: flex;
  align-items: center;
  cursor: pointer;
}

.visualization-container {
  position: relative;
  overflow: auto; /* Important for scrollbars if image is larger */
  flex-grow: 1; /* Container should take available space */
  background-color: #eee;
  display: flex; /* For centering image-container if needed */
  justify-content: center; /* Center image horizontally */
  align-items: flex-start; /* Align image to top, or center if preferred */
}

.image-container {
  position: relative; /* For absolute positioning of overlays */
  /* margin: auto; /* Centers the block element if parent is flex and aligns items center */
  /* Or remove margin: auto if visualization-container handles centering */
}


.manuscript-image {
  display: block; /* Removes extra space below image */
  max-width: 100%; /* Ensures image is responsive within its container */
  max-height: 100%; /* Ensures image is responsive within its container */
  user-select: none; /* Prevent image selection */
  opacity: 80%;
}

.placeholder-image {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #ddd;
  color: #666;
}

.points-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Allows clicks to pass through to SVG/image */
}

.point {
  position: absolute;
  width: 4px;
  height: 4px;
  background-color: rgba(255, 0, 0, 0.5);
  border-radius: 50%;
  transform: translate(-50%, -50%);
}

.graph-overlay {
  position: absolute;
  top: 0;
  left: 0;
  /* width and height are bound to scaledWidth/Height */
  cursor: default; /* Default cursor for SVG background */
}
.graph-overlay circle:hover, .graph-overlay line:hover {
    /* Optional: subtle hover effects if not handled by selection/key press states */
}

.processing-save-notice {
  position: fixed; /* Or absolute if you want it within a specific container */
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 20px 30px;
  border-radius: 8px;
  z-index: 10000; /* Ensure it's on top */
  font-size: 1.1em;
  text-align: center;
}





.loading {
  padding: 20px;
  text-align: center;
  font-style: italic;
  color: #666;
}

.error-message {
  padding: 10px; /* Reduced padding */
  background-color: #ffebee; /* Lighter red */
  color: #c62828; /* Darker red text */
  border: 1px solid #ef9a9a; /* Lighter red border */
  margin: 10px;
  border-radius: 4px;
  text-align: center;
}

.edit-controls {
  padding: 10px;
  background-color: #f9f9f9;
  border-top: 1px solid #ddd; /* Changed from bottom to top for typical layout */
  flex-shrink: 0; /* Controls should not shrink */
}

.edit-instructions {
  margin-bottom: 10px;
  font-size: 0.9em; /* Slightly smaller */
  color: #555;
}
.edit-instructions p {
  margin: 5px 0; /* Spacing for instruction paragraphs */
}


.edit-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 10px; /* Added margin for spacing from log */
}

button {
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid #ccc;
  background-color: #fff;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.2s ease;
}

button:hover:not(:disabled) {
  background-color: #e0e0e0; /* Darker hover */
}

button:disabled {
  opacity: 0.6; /* More visible disabled state */
  cursor: not-allowed;
  background-color: #f5f5f5; /* Lighter bg for disabled */
}

.modifications-log-container {
  padding: 10px;
  background-color: #f0f0f0; /* Slightly different background */
  border-top: 1px solid #ddd;
  flex-shrink: 0; /* Log should not shrink */
}
.modifications-log-container > button { /* Style for the main Save Graph button */
    margin-bottom: 10px;
}


.modifications-details h3 {
  font-size: 1.1em; /* Slightly larger */
  margin-top: 0; /* Remove top margin if it's the first element */
  margin-bottom: 8px;
  color: #333;
}
.modifications-details > button { /* Style for Reset All Changes button */
    margin-bottom: 10px;
}


.modifications-details ul {
  list-style-type: none;
  padding: 0;
  max-height: 150px; /* Limit height and make scrollable if too many */
  overflow-y: auto;
  border: 1px solid #ddd;
  background-color: #fff;
  border-radius: 3px;
}

.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px; /* Adjusted padding */
  border-bottom: 1px solid #eee;
  font-size: 0.85em; /* Smaller font for list items */
}
.modification-item:last-child {
  border-bottom: none;
}


.undo-button {
  font-size: 0.9em; /* Relative to parent button size */
  padding: 3px 8px; /* Smaller padding */
  background-color: #fffde7; /* Light yellow */
  border-color: #fff59d; /* Yellow border */
}
.undo-button:hover:not(:disabled) {
  background-color: #fff9c4; /* Darker yellow hover */
}
</style>