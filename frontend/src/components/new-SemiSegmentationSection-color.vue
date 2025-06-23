<template>
  <div class="manuscript-viewer">
    <!-- ... toolbar, edit-instructions-bar ... (no changes here) -->
    <div class="toolbar">
      <div class="main-controls">
        <h10>{{ manuscriptName }} - Page {{ currentPage }}</h10>
        <button @click="previousPage" :disabled="loading || isProcessingSave">Prev</button>
        <button @click="nextPage" :disabled="loading || isProcessingSave">Next</button>
        <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">Annotate Text</button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="editMode" :disabled="isProcessingSave" />
            Edit Mode
          </label>
        </div>
      </div>
      <div class="legend-area">
        <div class="legend-container">
          <div class="legend">
            <h4>Nodes (#edges)</h4>
            <ul>
              <li v-for="item in nodeLegendItems" :key="`node-legend-${item.value}`">
                <span class="color-box" :style="{ backgroundColor: item.color }"></span>
                {{ item.value }}
              </li>
            </ul>
          </div>
          <div class="legend">
            <h4>Edges (#overlaps)</h4>
            <ul>
              <li v-for="item in edgeLegendItems" :key="`edge-legend-${item.value}`">
                <span class="color-box" :style="{ backgroundColor: item.color }"></span>
                {{ item.value }}
              </li>
            </ul>
            <p v-if="modifications.length > 0 || workingGraph.edges.some(e => e.modified)" class="legend-modified-edge-note">
                <span class="color-box" :style="{ backgroundColor: MODIFIED_EDGE_COLOR }"></span>
                Modified
            </p>
          </div>
        </div>
      </div>
    </div>
     <div class="edit-instructions-bar" v-if="editMode">
        <p v-if="selectedNodes.length === 0 && !isAKeyPressed && !isDKeyPressed">Hold 'a' & hover to connect. Hold 'd' & hover to delete edges.</p>
        <p v-else-if="selectedNodes.length === 1 && !isAKeyPressed && !isDKeyPressed">Click 2nd node for edge, or background/another node to change.</p>
        <p v-else-if="selectedNodes.length === 2 && !isAKeyPressed && !isDKeyPressed">Click "Add/Delete Edge" below, or background/another node to change.</p>
        <p v-if="isAKeyPressed">Release 'A' to connect hovered nodes (MST).</p>
        <p v-if="isDKeyPressed">Hover over edges to delete. Release 'D' to stop.</p>
    </div>


    <div v-if="isProcessingSave" class="processing-save-notice">
      Saving graph and processing... Please wait.
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="loading" class="loading">
      Loading page data...
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
          <template v-for="(edge, index) in workingGraph.edges" :key="`edge-group-${index}-${edge.source}-${edge.target}`">
            <line
              v-if="getNodeById(edge.source) && getNodeById(edge.target)"
              :x1="scaleX(getNodeById(edge.source).x)"
              :y1="scaleY(getNodeById(edge.source).y)"
              :x2="scaleX(getNodeById(edge.target).x)"
              :y2="scaleY(getNodeById(edge.target).y)"
              :stroke="getEdgeColor(edge)"
              :stroke-width="isEdgeSelected(edge) ? 3 : 2.5"
              :stroke-opacity="1"
              @click.stop="editMode && onEdgeClick(edge, $event)"
            />
          </template>
          
          <circle
            v-for="(node) in workingGraph.nodes"
            :key="`node-${node.id}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(node.id)"
            :fill="getNodeColor(node.id)"
            :fill-opacity="1"
            @click.stop="editMode && onNodeClick(node.id, $event)"
          />
          
          <line
            v-if="editMode && selectedNodes.length === 1 && tempEndPoint && !isAKeyPressed && !isDKeyPressed && getNodeById(selectedNodes[0])"
            :x1="scaleX(getNodeById(selectedNodes[0]).x)"
            :y1="scaleY(getNodeById(selectedNodes[0]).y)"
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

    <!-- ... edit-actions-bar, modifications-log-container ... (no changes here) -->
    <!-- <div v-if="editMode && !isAKeyPressed && !isDKeyPressed" class="edit-actions-bar">
        <button @click="resetSelection">Cancel Sel.</button>
        <button 
          @click="addEdge" 
          :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
        >Add Edge</button>
        <button 
          @click="deleteEdge" 
          :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
        >Delete Edge</button>
    </div> -->

    <div v-if="editMode && graphIsLoaded" class="modifications-log-container">
        <button @click="saveModificationsAndStay" :disabled="isProcessingSave || modifications.length === 0">Save Graph</button>
        <div v-if="modifications.length > 0" class="modifications-details">
            <span>Modifications ({{ modifications.length }}) </span>
            <button @click="resetModifications" :disabled="isProcessingSave">Reset Changes</button>
            <ul>
              <li v-for="(mod, index) in modifications" :key="index" class="modification-item">
                {{ mod.type === 'add' ? 'Add' : 'Del' }} edge ({{ mod.source }} <-> {{ mod.target }})
                <button @click="undoModification(index)" class="undo-button">Undo</button>
              </li>
            </ul>
        </div>
        <p v-else-if="!loading && workingGraph.nodes && workingGraph.nodes.length > 0 && modifications.length === 0">No unsaved changes.</p>
    </div>
  </div>
</template>

<script setup>
// ... all script content remains the same as your last provided version
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue';
import { useAnnotationStore } from '@/stores/annotationStore';
import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js';
import { useRouter } from 'vue-router';

const router = useRouter();
const annotationStore = useAnnotationStore();

const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
const currentPage = computed(() => annotationStore.currentPage);
const isProcessingSave = ref(false);
const loading = ref(true);
const error = ref(null);
const dimensions = ref([0, 0]);
const points = ref([]);
const graph = ref({ nodes: [], edges: [] }); 
const imageData = ref('');
const imageLoaded = ref(false);

const editMode = ref(true);
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
const svgOverlayRef = ref(null);

const isDKeyPressed = ref(false);
const isAKeyPressed = ref(false);
const hoveredNodesForMST = reactive(new Set());
const NODE_HOVER_RADIUS = 20; 
const EDGE_HOVER_THRESHOLD = 5;

const effectiveShowPoints = computed(() => editMode.value);
const effectiveShowGraph = computed(() => editMode.value);
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0);

const DISTINCT_COLORS = [
  '#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000',
  '#000075', '#a9a9a9', '#000000'
];
const MODIFIED_EDGE_COLOR = '#e6194B'; 
const DEFAULT_NODE_COLOR = '#bfef45'; 
const DEFAULT_EDGE_COLOR = '#42d4f4'; 
const SELECTED_NODE_COLOR = '#ff9500'; 
const HOVER_COLLECT_NODE_COLOR = '#00bcd4'; 


const nodeColorsByDegree = reactive({});
const edgeColorsByOverlap = reactive({});
let assignedNodeColorCount = 0;
let assignedEdgeColorCount = 0;

function getColorForValue(value, type) {
  let mapping, assignedCounterRef, defaultColor; // Corrected: assignedCounterRef was assignedCounter before
  if (type === 'node') {
    mapping = nodeColorsByDegree;
    assignedCounterRef = assignedNodeColorCount; // Store the primitive value itself for modification
    defaultColor = DEFAULT_NODE_COLOR;
  } else { 
    mapping = edgeColorsByOverlap;
    assignedCounterRef = assignedEdgeColorCount; // Store the primitive value itself
    defaultColor = DEFAULT_EDGE_COLOR;
  }

  if (typeof value === 'undefined' || value === null) return defaultColor;

  if (mapping[value] === undefined) {
    if ((type === 'node' ? assignedNodeColorCount : assignedEdgeColorCount) < DISTINCT_COLORS.length) {
      mapping[value] = DISTINCT_COLORS[(type === 'node' ? assignedNodeColorCount : assignedEdgeColorCount)];
      if (type === 'node') assignedNodeColorCount++; else assignedEdgeColorCount++;
    } else {
      mapping[value] = DISTINCT_COLORS[value % DISTINCT_COLORS.length]; 
    }
  }
  return mapping[value];
}


const getNodeColor = (nodeId) => {
  const node = workingGraph.nodes.find(n => n.id === nodeId);
  if (!node) return DEFAULT_NODE_COLOR;

  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeId)) return HOVER_COLLECT_NODE_COLOR;
  if (isNodeSelected(nodeId)) return SELECTED_NODE_COLOR;

  return getColorForValue(node.numEdges, 'node');
};

const getEdgeColor = (edge) => {
  if (edge.modified) return MODIFIED_EDGE_COLOR;
  return getColorForValue(edge.overlaps, 'edge');
};

const nodeLegendItems = computed(() => {
  return Object.entries(nodeColorsByDegree)
    .map(([value, color]) => ({ value: parseInt(value), color }))
    .sort((a,b) => a.value - b.value);
});

const edgeLegendItems = computed(() => {
  return Object.entries(edgeColorsByOverlap)
    .map(([value, color]) => ({ value: parseInt(value), color }))
    .sort((a,b) => a.value - b.value);
});

function resetColorMappingsAndPopulateLegends() {
  Object.keys(nodeColorsByDegree).forEach(key => delete nodeColorsByDegree[key]);
  Object.keys(edgeColorsByOverlap).forEach(key => delete edgeColorsByOverlap[key]);
  assignedNodeColorCount = 0;
  assignedEdgeColorCount = 0;

  if (workingGraph.nodes) {
    const uniqueDegrees = new Set(workingGraph.nodes.map(n => n.numEdges).filter(d => typeof d !== 'undefined'));
    Array.from(uniqueDegrees).sort((a,b)=>a-b).forEach(degree => getColorForValue(degree, 'node'));
  }
  if (workingGraph.edges) {
    const uniqueOverlaps = new Set(workingGraph.edges.filter(e => !e.modified).map(e => e.overlaps).filter(o => typeof o !== 'undefined'));
     Array.from(uniqueOverlaps).sort((a,b)=>a-b).forEach(overlap => getColorForValue(overlap, 'edge'));
  }
}

const getNodeById = (id) => workingGraph.nodes.find(n => n.id === id);

const goToIMG2TXTPage = async () => {
  if (isProcessingSave.value) return;
  if (editMode.value && graphIsLoaded.value && modifications.value.length > 0) {
    isProcessingSave.value = true;
    try {
      await saveModifications();
      router.push({ name: 'img-2-txt' });
    } catch (err) {
      alert(`Error saving graph: ${err.message}. Cannot proceed to Annotate Text.`);
    } finally {
      isProcessingSave.value = false;
    }
  } else {
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
  modifications.value = [];
  
  try {
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + `/semi-segment/${manuscriptName.value}/${currentPage.value}`
    );
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data');
    const data = await response.json();
    
    updateCanvasSize(data.dimensions[0], data.dimensions[1]);
    points.value = data.points.map(point => ({ coordinates: [point[0], point[1]], segment: null }));
    
    if (data.graph && data.graph.nodes && data.graph.nodes.length > 0) {
      data.graph.nodes.forEach(node => {
        if (typeof node.numEdges === 'undefined') { 
            let count = 0;
            if (data.graph.edges) {
                for (const edge of data.graph.edges) {
                    if (edge.source === node.id || edge.target === node.id) count++;
                }
            }
            node.numEdges = count;
        }
      });
      data.graph.edges.forEach(edge => {
        if (typeof edge.overlaps === 'undefined') edge.overlaps = 1; 
        if (typeof edge.label === 'undefined') edge.label = 0;
      });
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
  workingGraph.edges.forEach(edge => {
    if (edge.modified === undefined) {
      edge.modified = false; 
    }
  });
  resetSelection();
  resetColorMappingsAndPopulateLegends();
};

const resetSelection = () => { selectedNodes.value = []; tempEndPoint.value = null;};
const onNodeClick = (nodeId, event) => {   if (isAKeyPressed.value || isDKeyPressed.value) return; event.stopPropagation(); const existingIndex = selectedNodes.value.indexOf(nodeId); if (existingIndex !== -1) { selectedNodes.value.splice(existingIndex, 1); } else { if (selectedNodes.value.length < 2) { selectedNodes.value.push(nodeId); } else { selectedNodes.value = [nodeId]; } } tempEndPoint.value = null;};
const onEdgeClick = (edge, event) => {   if (isAKeyPressed.value || isDKeyPressed.value) return; event.stopPropagation(); selectedNodes.value = [edge.source, edge.target];};
const onBackgroundClick = () => {   if (isAKeyPressed.value || isDKeyPressed.value) return; resetSelection();};
const edgeExists = (nodeAId, nodeBId) => {   if (nodeAId === undefined || nodeBId === undefined) return false; return workingGraph.edges.some(e => (e.source === nodeAId && e.target === nodeBId) || (e.source === nodeBId && e.target === nodeAId));};

const addEdgeManual = () => {
  if (selectedNodes.value.length !== 2) return;
  const [sourceId, targetId] = selectedNodes.value;
  if (sourceId === targetId || edgeExists(sourceId, targetId)) return;
  
  const newEdge = { 
    source: sourceId, 
    target: targetId, 
    overlaps: 1, 
    modified: true,
    label: 0 
  };
  workingGraph.edges.push(newEdge);

  const sourceNode = getNodeById(sourceId);
  const targetNode = getNodeById(targetId);
  if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1;
  if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1;
  
  modifications.value.push({ type: 'add', ...newEdge });
  resetColorMappingsAndPopulateLegends();
  resetSelection();
};
const addEdge = addEdgeManual;

const deleteEdgeManual = () => {
  if (selectedNodes.value.length !== 2) return;
  const [sourceId, targetId] = selectedNodes.value;
  const edgeIndex = workingGraph.edges.findIndex(e => 
    (e.source === sourceId && e.target === targetId) || (e.source === targetId && e.target === sourceId)
  );
  if (edgeIndex === -1) return;
  
  const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0];

  const sourceNode = getNodeById(removedEdge.source);
  const targetNode = getNodeById(removedEdge.target);
  if (sourceNode && sourceNode.numEdges > 0) sourceNode.numEdges--;
  if (targetNode && targetNode.numEdges > 0) targetNode.numEdges--;

  modifications.value.push({ 
    type: 'delete', 
    source: removedEdge.source, 
    target: removedEdge.target, 
    overlaps: removedEdge.overlaps,
    label: removedEdge.label !== undefined ? removedEdge.label : 0 
  });
  resetColorMappingsAndPopulateLegends();
  resetSelection();
};
const deleteEdge = deleteEdgeManual;

const undoModification = (index) => {
  const mod = modifications.value[index];
  if (mod.type === 'add') {
    const edgeIndex = workingGraph.edges.findIndex(e => 
      e.source === mod.source && e.target === mod.target && e.modified
    );
    if (edgeIndex !== -1) {
      workingGraph.edges.splice(edgeIndex, 1);
      const sourceNode = getNodeById(mod.source);
      const targetNode = getNodeById(mod.target);
      if (sourceNode && sourceNode.numEdges > 0) sourceNode.numEdges--;
      if (targetNode && targetNode.numEdges > 0) targetNode.numEdges--;
    }
  } else if (mod.type === 'delete') {
    const reAddedEdge = { 
        source: mod.source, 
        target: mod.target, 
        overlaps: mod.overlaps,
        label: mod.label, 
        modified: true 
    };
    workingGraph.edges.push(reAddedEdge);
    const sourceNode = getNodeById(mod.source);
    const targetNode = getNodeById(mod.target);
    if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1;
    if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1;
  }
  modifications.value.splice(index, 1);
  resetColorMappingsAndPopulateLegends();
};

const resetModifications = () => {
  resetWorkingGraph(); 
  modifications.value = [];
};

const isNodeSelected = (nodeId) => selectedNodes.value.includes(nodeId);
const isEdgeSelected = (edge) => {   return selectedNodes.value.length === 2 && ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) || (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));};
const getNodeRadius = (nodeId) => {   if (isAKeyPressed.value && hoveredNodesForMST.has(nodeId)) return 5; return isNodeSelected(nodeId) ? 6 : 3;};
const confirmAndNavigate = async (navigationAction) => {   if (isProcessingSave.value) { alert("Please wait for the current save operation to complete."); return; } if (modifications.value.length > 0) { if (confirm('You have unsaved changes. Do you want to save them before navigating?')) { isProcessingSave.value = true; try { await saveModifications(); modifications.value = []; navigationAction(); } catch (err) { alert("Failed to save changes. Please try again or discard changes to navigate."); } finally { isProcessingSave.value = false; } } else { modifications.value = []; navigationAction(); } } else { navigationAction(); }};
const nextPage = () => confirmAndNavigate(() => annotationStore.nextPage());
const previousPage = () => confirmAndNavigate(() => annotationStore.previousPage());
const handleGlobalKeyDown = (e) => {   if (e.key.toLowerCase() === 'e' && !e.ctrlKey && !e.metaKey) { if (isProcessingSave.value) return; e.preventDefault(); editMode.value = !editMode.value; return; } if (e.key.toLowerCase() === 't' && !e.ctrlKey && !e.metaKey) { if (loading.value || isProcessingSave.value) return; e.preventDefault(); goToIMG2TXTPage(); return; } if (!editMode.value || e.repeat) return; if (e.key.toLowerCase() === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); } if (e.key.toLowerCase() === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); }};
const handleGlobalKeyUp = (e) => {   if (!editMode.value) return; if (e.key.toLowerCase() === 'd') isDKeyPressed.value = false; if (e.key.toLowerCase() === 'a') { isAKeyPressed.value = false; if (hoveredNodesForMST.size >= 2) addMSTEdges(); hoveredNodesForMST.clear(); }};
const handleSvgMouseMove = (event) => {   if (!editMode.value || !svgOverlayRef.value) return; const svgRect = svgOverlayRef.value.getBoundingClientRect(); const mouseX = event.clientX - svgRect.left; const mouseY = event.clientY - svgRect.top; if (isDKeyPressed.value) handleEdgeHoverDelete(mouseX, mouseY); else if (isAKeyPressed.value) handleNodeHoverCollect(mouseX, mouseY); else if (selectedNodes.value.length === 1 && getNodeById(selectedNodes.value[0])) tempEndPoint.value = { x: mouseX, y: mouseY }; else tempEndPoint.value = null; };
const handleSvgMouseLeave = () => {   if (selectedNodes.value.length === 1 && !isAKeyPressed.value && !isDKeyPressed.value) { tempEndPoint.value = null; }};
function distanceToLineSegment(px, py, x1, y1, x2, y2) {   const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2; if (l2 === 0) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2); let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2; t = Math.max(0, Math.min(1, t)); const projX = x1 + t * (x2 - x1); const projY = y1 + t * (y2 - y1); return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);};

const handleEdgeHoverDelete = (mouseX, mouseY) => {
  let edgeRemoved = false;
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i];
    const nodeSource = getNodeById(edge.source);
    const nodeTarget = getNodeById(edge.target);
    if (!nodeSource || !nodeTarget) continue;

    const x1 = scaleX(nodeSource.x);
    const y1 = scaleY(nodeSource.y);
    const x2 = scaleX(nodeTarget.x);
    const y2 = scaleY(nodeTarget.y);
    const dist = distanceToLineSegment(mouseX, mouseY, x1, y1, x2, y2);

    if (dist < EDGE_HOVER_THRESHOLD) {
      const removedEdge = workingGraph.edges.splice(i, 1)[0];
      if (nodeSource && nodeSource.numEdges > 0) nodeSource.numEdges--;
      if (nodeTarget && nodeTarget.numEdges > 0) nodeTarget.numEdges--;
      
      modifications.value.push({
        type: 'delete', source: removedEdge.source, target: removedEdge.target, 
        overlaps: removedEdge.overlaps, 
        label: removedEdge.label !== undefined ? removedEdge.label : 0
      });
      edgeRemoved = true;
    }
  }
  if (edgeRemoved) resetColorMappingsAndPopulateLegends();
};

const handleNodeHoverCollect = (mouseX, mouseY) => {   workingGraph.nodes.forEach(node => { const nodeX = scaleX(node.x); const nodeY = scaleY(node.y); const distSq = (mouseX - nodeX) ** 2 + (mouseY - nodeY) ** 2; if (distSq < (NODE_HOVER_RADIUS / 2) ** 2) { hoveredNodesForMST.add(node.id); } });};
class DSU {   constructor() { this.parent = {}; } init(nodeIndices) { this.parent = {}; nodeIndices.forEach(idx => this.parent[idx] = idx); } find(i) { if (this.parent[i] === i) return i; return this.parent[i] = this.find(this.parent[i]); } union(i, j) { const rootI = this.find(i); const rootJ = this.find(j); if (rootI !== rootJ) { this.parent[rootJ] = rootI; return true; } return false; }};
function calculateMST(nodeIds, allNodesData) {   if (nodeIds.length < 2) return []; const nodesForMST = nodeIds.map(id => allNodesData.find(n => n.id === id)).filter(n => n); if (nodesForMST.length < 2) return []; const mstEdges = []; const potentialEdges = []; for (let i = 0; i < nodesForMST.length; i++) { for (let j = i + 1; j < nodesForMST.length; j++) { const p1 = nodesForMST[i]; const p2 = nodesForMST[j]; const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2); potentialEdges.push({ source: p1.id, target: p2.id, weight: dist }); } } potentialEdges.sort((a, b) => a.weight - b.weight); const dsu = new DSU(); dsu.init(nodeIds); for (const edge of potentialEdges) { if (dsu.union(edge.source, edge.target)) { mstEdges.push({ source: edge.source, target: edge.target }); } } return mstEdges;};

const addMSTEdges = () => {
  const nodesToConnect = Array.from(hoveredNodesForMST);
  if (nodesToConnect.length < 2) return;

  const mstNewEdges = calculateMST(nodesToConnect, workingGraph.nodes);
  let edgeAdded = false;
  mstNewEdges.forEach(edge => {
    if (!edgeExists(edge.source, edge.target)) {
      const newEdgeData = {
        source: edge.source, target: edge.target,
        overlaps: 1, 
        modified: true,
        label: 0 
      };
      workingGraph.edges.push(newEdgeData);
      
      const sourceNode = getNodeById(edge.source);
      const targetNode = getNodeById(edge.target);
      if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1;
      if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1;
      
      modifications.value.push({ type: 'add', ...newEdgeData });
      edgeAdded = true;
    }
  });
  if (edgeAdded) resetColorMappingsAndPopulateLegends();
};

watch(() => annotationStore.currentPage, (newPage, oldPage) => {   if (isProcessingSave.value) return; if (newPage && newPage !== oldPage) { fetchPageData(); } else if (!newPage && oldPage) { points.value = []; graph.value = { nodes: [], edges: [] }; modifications.value = []; resetWorkingGraph(); loading.value = false; error.value = null; }}, { immediate: true });
watch(editMode, (newValue) => {   if (!newValue) { resetSelection(); isAKeyPressed.value = false; isDKeyPressed.value = false; hoveredNodesForMST.clear(); tempEndPoint.value = null; }});
onMounted(() => {   window.addEventListener('keydown', handleGlobalKeyDown); window.addEventListener('keyup', handleGlobalKeyUp); if (annotationStore.currentPage && !imageLoaded.value && !loading.value && !isProcessingSave.value) { fetchPageData(); }});
onBeforeUnmount(() => {   window.removeEventListener('keydown', handleGlobalKeyDown); window.removeEventListener('keyup', handleGlobalKeyUp);});

const saveModifications = async () => {
   try {
    console.log('Saving modifications...');
    const request = {
      graph: workingGraph,
      modifications: modifications.value,
      points: points.value.map(point => point.segment),
      modelName: annotationStore.modelName
    };
    
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Failed to parse error from backend" }));
      throw new Error(errorData.error || 'Failed to save and process on backend');
    }
    const responseData = await response.json();
    
    graph.value = JSON.parse(JSON.stringify(workingGraph));
    workingGraph.edges.forEach(edge => edge.modified = false);
    modifications.value = [];
    error.value = null;
    
    if (responseData.lines) {
      if (!annotationStore.recognitions[manuscriptName.value]) {
        annotationStore.recognitions[manuscriptName.value] = {};
      }
      annotationStore.recognitions[manuscriptName.value][currentPage.value] = responseData.lines;
    }
    resetColorMappingsAndPopulateLegends(); 
    console.log('Graph saved and page processed successfully.');

  } catch (err) {
    console.error('Error in saveModifications:', err);
    error.value = err.message || 'Failed to save modifications';
    throw err; 
  }
};

const saveModificationsAndStay = async () => {   if (isProcessingSave.value) return; isProcessingSave.value = true; try { await saveModifications(); alert("Graph saved successfully!"); } catch (err) { alert(`Failed to save graph: ${err.message}`); } finally { isProcessingSave.value = false; }};

</script>

<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  font-size: 0.9rem;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start; 
  padding: 3px 8px; 
  background-color: #f0f0f0;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
  gap: 10px;
}
.main-controls {
  display: flex;
  align-items: center;
  gap: 8px; 
  flex-wrap: nowrap; 
}
.main-controls h10 {
  font-size: 0.9em; 
  white-space: nowrap;
}
.main-controls button {
  padding: 3px 8px; 
  font-size: 0.85em;
}
.toggle-container label {
  font-size: 0.85em;
}

.legend-area {
  margin-left: auto; 
}
.legend-container {
  display: flex;
  flex-direction: row; 
  gap: 8px;
  padding: 3px;
  font-size: 0.75em; 
  background-color: #f9f9f9;
  border-radius: 3px;
  border: 1px solid #e0e0e0;
}
.legend h4 {
  margin-top: 0;
  margin-bottom: 3px;
  font-size: 0.9em; 
  font-weight: bold;
}
.legend ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
}
.legend li {
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 1px 3px;
  border: 1px solid #eee;
  border-radius: 2px;
  background-color: #fff;
}
.color-box {
  width: 10px;
  height: 10px;
  border: 1px solid #ccc;
  display: inline-block;
}
.legend-modified-edge-note {
    margin-top: 2px;
    display: flex;
    align-items: center;
    gap: 3px;
    font-style: italic;
    font-size: 0.9em;
}
.edit-instructions-bar {
  background-color: #e9ecef; 
  padding: 3px 8px;
  font-size: 0.8em;
  color: #495057;
  border-bottom: 1px solid #ddd;
  text-align: center;
  flex-shrink: 0;
}
.edit-instructions-bar p {
  margin: 0;
  display: inline; 
  margin-right: 10px; 
}

.processing-save-notice, .loading, .error-message { 
  padding: 15px; text-align: center; flex-shrink: 0;
}
.processing-save-notice { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: rgba(0,0,0,0.75); color: white; border-radius: 8px; z-index: 10000; font-size: 1em; }
.loading { font-style: italic; color: #666; flex-grow: 1; display:flex; align-items:center; justify-content:center;}
.error-message { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; margin: 5px; border-radius: 4px; }

.visualization-container {
  position: relative;
  overflow: auto; 
  flex-grow: 1; 
  background-color: #e0e0e0; 
  display: flex;
  justify-content: center;
  align-items: center; 
}

.image-container {
  position: relative; 
  /* width and height are bound via :style to scaledWidth/Height */
  /* REMOVED max-width: 100%; and max-height: 100%; to allow container to be its explicit scaled size */
  /* This will make .visualization-container provide scrollbars if image-container is larger */
}

.manuscript-image {
  display: block; 
  /* width and height attributes are set to scaledWidth/Height by binding */
  /* max-width/max-height 100% here means 100% of image-container, which is now correctly sized */
  max-width: 100%; 
  max-height: 100%;
  user-select: none;
  opacity: 0.85; 
  object-fit: contain; 
}
.placeholder-image {   display: flex; align-items: center; justify-content: center; background-color: #ddd; color: #666; }
.points-overlay {   position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
.point {   position: absolute; width: 4px; height: 4px; background-color: rgba(255, 0, 0, 0.5); border-radius: 50%; transform: translate(-50%, -50%); }
.graph-overlay {   position: absolute; top: 0; left: 0; cursor: default; }

.edit-actions-bar {
  padding: 5px 8px;
  background-color: #f8f9fa;
  border-top: 1px solid #ddd;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.edit-actions-bar button {
  padding: 3px 8px;
  font-size: 0.85em;
}

.modifications-log-container {
  padding: 5px 8px;
  background-color: #f0f0f0;
  border-top: 1px solid #ddd;
  flex-shrink: 0;
  display: flex; 
  align-items: center;
  gap: 10px;
  font-size: 0.8em;
}
.modifications-log-container > button { 
    padding: 3px 8px;
    font-size: 0.9em; 
}
.modifications-details {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-grow: 1; 
}
.modifications-details span { 
 white-space: nowrap;
}
.modifications-details > button { 
    padding: 2px 6px;
    font-size: 0.9em;
    margin-left: 5px; 
}
.modifications-details ul {
  list-style-type: none;
  padding: 0 3px;
  margin: 0;
  max-height: 40px; 
  overflow-y: auto;
  border: 1px solid #ddd;
  background-color: #fff;
  border-radius: 3px;
  flex-grow: 1; 
  display: flex; 
  flex-direction: column;
}
.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 4px;
  border-bottom: 1px solid #eee;
  font-size: 0.9em; 
  white-space: nowrap;
}
.modification-item:last-child { border-bottom: none; }
.undo-button {
  font-size: 0.9em;
  padding: 1px 4px;
  background-color: #fffde7;
  border-color: #fff59d;
  margin-left: 5px;
}

button:disabled { 
  opacity: 0.5;
  cursor: not-allowed;
  background-color: #e9ecef;
}
button:hover:not(:disabled) {
  background-color: #dde1e6;
}
</style>