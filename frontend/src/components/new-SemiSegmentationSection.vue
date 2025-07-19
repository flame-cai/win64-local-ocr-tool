<template>
  <div class="manuscript-viewer">
    <!-- Toolbar and Edit Instructions Bar -->
    <div class="toolbar">
      <div class="main-controls">
        <h10>{{ manuscriptName }} - Page {{ currentPage }}</h10>
        <!-- ADDED THE SAVE IMAGE BUTTON HERE -->
        <button @click="saveImageWithOverlay" :disabled="loading || isProcessingSave || !imageLoaded">Save Image</button>
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
    <div v-if="error" class="error-message"> {{ error }} </div>
    <div v-if="loading" class="loading"> Loading page data... </div>

    <div 
      v-else 
      class="visualization-container" 
      ref="visualizationContainerRef"
      @wheel.prevent="handleWheelZoom"
    >
      <div 
        class="transform-wrapper"
        ref="transformWrapperRef"
        :style="transformStyle"
        @mousedown="handlePanStart"
      >
        <div 
          class="image-wrapper" 
          :style="{ width: `${imageOriginalWidth}px`, height: `${imageOriginalHeight}px` }"
        >
          <img 
            v-if="imageData" 
            :src="`data:image/jpeg;base64,${imageData}`" 
            :width="imageOriginalWidth" 
            :height="imageOriginalHeight" 
            class="manuscript-image"
            alt="Manuscript Page"
            @load="onImageLoad"
          />
          <div v-else class="placeholder-image" :style="{ width: `${imageOriginalWidth}px`, height: `${imageOriginalHeight}px` }">
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
                left: `${point.coordinates[0]}px`,
                top: `${point.coordinates[1]}px`,
                width: `${BASE_POINT_VISUAL_SIZE / currentScale}px`,
                height: `${BASE_POINT_VISUAL_SIZE / currentScale}px`,
                transform: 'translate(-50%, -50%)'
              }"
              :title="`Point ${index}: (${point.coordinates[0]}, ${point.coordinates[1]})`"
            ></div>
          </div>
        </div>
        
        <svg 
          v-if="effectiveShowGraph && workingGraph.nodes && workingGraph.nodes.length > 0 && imageOriginalWidth > 0 && imageOriginalHeight > 0" 
          class="graph-overlay"
          :width="imageOriginalWidth"
          :height="imageOriginalHeight"
          @click="handleSvgBackgroundClick"
          @mousemove="handleSvgMouseMove"
          @mouseleave="handleSvgMouseLeave"
          ref="svgOverlayRef"
        >
          <!-- Edges drawn first -->
          <template v-for="(edge, index) in workingGraph.edges" :key="`edge-group-${index}-${edge.source}-${edge.target}`">
            <line
              v-if="getNodeById(edge.source) && getNodeById(edge.target)"
              :x1="getNodeById(edge.source).x"
              :y1="getNodeById(edge.source).y"
              :x2="getNodeById(edge.target).x"
              :y2="getNodeById(edge.target).y"
              :stroke="getEdgeColor(edge)"
              :stroke-width="getDynamicStrokeWidth(edge)"
              :stroke-opacity="1"
              @click.stop="editMode && onEdgeClick(edge, $event)"
            />
          </template>
          
          <!-- Nodes drawn on top of edges -->
          <circle
            v-for="(node) in workingGraph.nodes"
            :key="`node-${node.id}`"
            :cx="node.x"
            :cy="node.y"
            :r="getDynamicNodeRadius(node.id)"
            :fill="getNodeColor(node.id)"
            :fill-opacity="1"
            @click.stop="editMode && onNodeClick(node.id, $event)"
          />
          
          <!-- Temporary line for drawing new edge -->
          <line
            v-if="editMode && selectedNodes.length === 1 && tempLineEndPoint && !isAKeyPressed && !isDKeyPressed && getNodeById(selectedNodes[0])"
            :x1="getNodeById(selectedNodes[0]).x"
            :y1="getNodeById(selectedNodes[0]).y"
            :x2="tempLineEndPoint.x" 
            :y2="tempLineEndPoint.y"
            stroke="#ff9500"
            :stroke-width="BASE_TEMP_LINE_STROKE_WIDTH / currentScale"
            stroke-dasharray="5,5"
            stroke-opacity="1"
          />
        </svg>
      </div>
    </div>

    <!-- Edit Actions Bar & Modifications Log -->
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
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive, nextTick } from 'vue';
import { useAnnotationStore } from '@/stores/annotationStore';
import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js';
import { useRouter } from 'vue-router';

const router = useRouter();
const annotationStore = useAnnotationStore();

// --- STATE FOR VISUALIZATION & INTERACTION ---
const visualizationContainerRef = ref(null);
const transformWrapperRef = ref(null);
const svgOverlayRef = ref(null);

const imageOriginalWidth = ref(0);
const imageOriginalHeight = ref(0);
const imageData = ref('');
const imageLoaded = ref(false);

const currentScale = ref(1.0);
const translateX = ref(0);
const translateY = ref(0);

const isPanning = ref(false);
const panStartX = ref(0);
const panStartY = ref(0);

const tempLineEndPoint = ref(null); // Stores {x, y} in graph coordinates for drawing temp line

// --- CONSTANTS FOR DYNAMIC SIZING ---
const BASE_NODE_RADIUS_SVG = 4; // Base radius in SVG units (graph coordinate system)
const SELECTED_NODE_BASE_RADIUS_SVG = 6;
const HOVER_COLLECT_NODE_BASE_RADIUS_SVG = 5;
const BASE_EDGE_STROKE_WIDTH_SVG = 3;
const SELECTED_EDGE_STROKE_WIDTH_SVG = 3;
const BASE_TEMP_LINE_STROKE_WIDTH = 2; // For the orange dashed line
const BASE_POINT_VISUAL_SIZE = 4; // Apparent visual size for points

// --- EXISTING STATE (mostly unchanged, some related to UI) ---
const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
const currentPage = computed(() => annotationStore.currentPage);
const isProcessingSave = ref(false);
const loading = ref(true);
const error = ref(null);
const points = ref([]);
const graph = ref({ nodes: [], edges: [] }); 
const workingGraph = reactive({ nodes: [], edges: [] }); 

const editMode = ref(true);
const selectedNodes = ref([]);
const modifications = ref([]);

const isDKeyPressed = ref(false);
const isAKeyPressed = ref(false);
const hoveredNodesForMST = reactive(new Set());
const NODE_HOVER_RADIUS_PX = 20; // Hover detection radius in screen pixels (diameter for original logic)
const EDGE_HOVER_THRESHOLD_PX = 5; // Hover detection threshold for edges in screen pixels

const effectiveShowPoints = computed(() => editMode.value);
const effectiveShowGraph = computed(() => editMode.value); 
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0);

// Colors and Legends (mostly unchanged)
const DISTINCT_COLORS = ['#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#000000'];
const MODIFIED_EDGE_COLOR = '#e6194B'; 
const DEFAULT_NODE_COLOR = '#bfef45'; 
const DEFAULT_EDGE_COLOR = '#42d4f4'; 
const SELECTED_NODE_COLOR = '#ff9500'; 
const HOVER_COLLECT_NODE_COLOR = '#00bcd4'; 
const nodeColorsByDegree = reactive({});
const edgeColorsByOverlap = reactive({});
let assignedNodeColorCount = 0;
let assignedEdgeColorCount = 0;


// --- COMPUTED PROPERTIES FOR TRANSFORMATION ---
const transformStyle = computed(() => {
  return {
    transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${currentScale.value})`,
    transformOrigin: '0 0' // Critical for scaling around 0,0 of the wrapper
  };
});

// --- IMAGE AND VIEWPORT HANDLING ---
const onImageLoad = (event) => {
  console.log('Image loaded event fired.');
  const img = event.target;
  if (img.naturalWidth > 0 && img.naturalHeight > 0) {
    imageOriginalWidth.value = img.naturalWidth;
    imageOriginalHeight.value = img.naturalHeight;
    imageLoaded.value = true;
    console.log(`Image dimensions set: ${imageOriginalWidth.value}x${imageOriginalHeight.value}. Triggering view reset.`);
    // Defer resetView to ensure container has dimensions
    nextTick(() => {
      resetView();
    });
  } else {
    console.warn('Image loaded but naturalWidth or naturalHeight is zero.');
  }
};

const resetView = () => {
  if (!visualizationContainerRef.value || imageOriginalWidth.value === 0 || imageOriginalHeight.value === 0) {
    console.log('Cannot reset view: visualization container or image dimensions not ready.', 
                { hasContainer: !!visualizationContainerRef.value, 
                  imgWidth: imageOriginalWidth.value, 
                  imgHeight: imageOriginalHeight.value });
    return;
  }
  console.log('Resetting view...');
  const containerEl = visualizationContainerRef.value;
  const containerWidth = containerEl.clientWidth;
  const containerHeight = containerEl.clientHeight;

  if (containerWidth === 0 || containerHeight === 0) {
    console.warn('Visualization container has zero dimensions. Cannot calculate fit.');
    // Default to a small scale and centered if container size is unknown
    currentScale.value = 0.5; 
    translateX.value = 0;
    translateY.value = 0;
    return;
  }

  const scaleToFitWidth = containerWidth / imageOriginalWidth.value;
  const scaleToFitHeight = containerHeight / imageOriginalHeight.value;
  
  // Fit image within container, but don't scale up beyond 100% initially
  currentScale.value = Math.min(scaleToFitWidth, scaleToFitHeight, 1.0); 
  if (currentScale.value <= 0) currentScale.value = 0.1; // Ensure scale is positive

  // Center the image
  translateX.value = (containerWidth - imageOriginalWidth.value * currentScale.value) / 2;
  translateY.value = (containerHeight - imageOriginalHeight.value * currentScale.value) / 2;
  
  console.log(`View reset: Scale=${currentScale.value.toFixed(3)}, TX=${translateX.value.toFixed(1)}px, TY=${translateY.value.toFixed(1)}px`);
};

// --- ZOOM HANDLING ---
const MIN_SCALE = 0.05;
const MAX_SCALE = 15.0;
const ZOOM_SENSITIVITY_FACTOR = 0.001;

const handleWheelZoom = (event) => {
  if (!visualizationContainerRef.value || !imageLoaded.value) return;
  event.preventDefault();

  const rect = visualizationContainerRef.value.getBoundingClientRect();
  const mouseX_viewport = event.clientX - rect.left; // Mouse X relative to container
  const mouseY_viewport = event.clientY - rect.top; // Mouse Y relative to container

  const oldScale = currentScale.value;
  const oldTx = translateX.value;
  const oldTy = translateY.value;

  const zoomAmount = event.deltaY * ZOOM_SENSITIVITY_FACTOR * -1; // Invert deltaY for natural zoom
  let newScale = oldScale * (1 + zoomAmount);
  newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));

  // Adjust translation to zoom around the mouse pointer
  // Formula: T_new = P_mouse_viewport - (P_mouse_viewport - T_old) * (S_new / S_old)
  translateX.value = mouseX_viewport - (mouseX_viewport - oldTx) * (newScale / oldScale);
  translateY.value = mouseY_viewport - (mouseY_viewport - oldTy) * (newScale / oldScale);
  currentScale.value = newScale;
};

// --- PAN HANDLING ---
const handlePanStart = (event) => {
  // Prevent panning if interacting with graph elements in edit mode or using A/D keys
  const targetTag = event.target.tagName?.toLowerCase();
  if (editMode.value && (targetTag === 'circle' || targetTag === 'line' || isAKeyPressed.value || isDKeyPressed.value)) {
    return;
  }
  // Allow panning only with left mouse button
  if (event.button !== 0) return;

  event.preventDefault();
  isPanning.value = true;
  panStartX.value = event.clientX;
  panStartY.value = event.clientY;
  document.addEventListener('mousemove', handlePanMove);
  document.addEventListener('mouseup', handlePanEnd);
  if (visualizationContainerRef.value) visualizationContainerRef.value.style.cursor = 'grabbing';
  console.log('Pan Start');
};

const handlePanMove = (event) => {
  if (!isPanning.value) return;
  event.preventDefault();
  const dx = event.clientX - panStartX.value;
  const dy = event.clientY - panStartY.value;

  translateX.value += dx;
  translateY.value += dy;

  panStartX.value = event.clientX;
  panStartY.value = event.clientY;
};

const handlePanEnd = (event) => {
  if (!isPanning.value) return;
  event.preventDefault();
  isPanning.value = false;
  document.removeEventListener('mousemove', handlePanMove);
  document.removeEventListener('mouseup', handlePanEnd);
  if (visualizationContainerRef.value) visualizationContainerRef.value.style.cursor = 'grab';
  console.log('Pan End');
};

// --- COORDINATE CONVERSION ---
// Converts mouse coordinates from viewport space to graph space (original image coordinates)
const convertViewportToGraphCoords = (viewportX, viewportY) => {
  const graphX = (viewportX - translateX.value) / currentScale.value;
  const graphY = (viewportY - translateY.value) / currentScale.value;
  return { x: graphX, y: graphY };
};

// --- DYNAMIC ELEMENT SIZING ---
// These functions calculate sizes in SVG units to maintain roughly constant apparent visual size on screen
const getDynamicNodeRadius = (nodeId) => {
  let baseRadiusSvgUnits = BASE_NODE_RADIUS_SVG;
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeId)) {
    baseRadiusSvgUnits = HOVER_COLLECT_NODE_BASE_RADIUS_SVG;
  } else if (isNodeSelected(nodeId)) {
    baseRadiusSvgUnits = SELECTED_NODE_BASE_RADIUS_SVG;
  }
  // To maintain apparent size: radius_svg = desired_pixel_radius / scale
  // We use baseRadiusSvgUnits as the "desired pixel radius" at scale=1
  const radius = baseRadiusSvgUnits / currentScale.value;
  return Math.max(1 / currentScale.value, radius); // Ensure minimum visible size
};

const getDynamicStrokeWidth = (edge) => {
  let baseStrokeWidthSvgUnits = isEdgeSelected(edge) ? SELECTED_EDGE_STROKE_WIDTH_SVG : BASE_EDGE_STROKE_WIDTH_SVG;
  const strokeWidth = baseStrokeWidthSvgUnits / currentScale.value;
  return Math.max(0.5 / currentScale.value, strokeWidth); // Ensure minimum visible stroke
};

// --- SVG INTERACTION HANDLERS (MOUSE MOVE, CLICKS) ---
const handleSvgMouseMove = (event) => {
  if (!editMode.value || !svgOverlayRef.value || !imageLoaded.value) return;
  
  const svgRect = svgOverlayRef.value.getBoundingClientRect();
  const mouseX_svgElement = event.clientX - svgRect.left;
  const mouseY_svgElement = event.clientY - svgRect.top;

  const mouseX_graph = mouseX_svgElement / currentScale.value;
  const mouseY_graph = mouseY_svgElement / currentScale.value;

  if (isDKeyPressed.value) {
    handleEdgeHoverDelete(mouseX_graph, mouseY_graph);
  } else if (isAKeyPressed.value) {
    handleNodeHoverCollect(mouseX_graph, mouseY_graph);
  } else if (selectedNodes.value.length === 1 && getNodeById(selectedNodes.value[0])) {
    tempLineEndPoint.value = { x: mouseX_graph, y: mouseY_graph };
  } else {
    tempLineEndPoint.value = null;
  }
};

const handleSvgMouseLeave = () => {
  if (selectedNodes.value.length === 1 && !isAKeyPressed.value && !isDKeyPressed.value) {
    tempLineEndPoint.value = null;
  }
};

const handleSvgBackgroundClick = (event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || isPanning.value) return;
  console.log('SVG background click');
  resetSelection();
};

// --- DATA FETCHING & INITIALIZATION (adapted) ---
const fetchPageData = async () => {
  if (!manuscriptName.value || !currentPage.value) return;
  console.log(`Fetching data for: ${manuscriptName.value} - Page ${currentPage.value}`);
  loading.value = true;
  error.value = null;
  points.value = [];
  graph.value = { nodes: [], edges: [] };
  imageData.value = '';
  imageLoaded.value = false;
  imageOriginalWidth.value = 0; // Reset for new image
  imageOriginalHeight.value = 0;
  modifications.value = [];
  
  currentScale.value = 1.0;
  translateX.value = 0;
  translateY.value = 0;
  
  try {
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + `/semi-segment/${manuscriptName.value}/${currentPage.value}`
    );
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data');
    const data = await response.json();
    
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

    if (data.image) {
      console.log('Image data received from backend.');
      imageData.value = data.image; 
    } else {
      console.warn('No image data received from backend.');
      imageLoaded.value = true; 
      nextTick(resetView);
    }

  } catch (err) {
    console.error('Error fetching page data:', err);
    error.value = err.message || 'Failed to load page data';
    imageLoaded.value = true;
  } finally {
    loading.value = false;
    console.log('Finished fetching page data.');
  }
};


// --- EXISTING GRAPH LOGIC ---
function getColorForValue(value, type) {
  let mapping, assignedCounter, defaultColor;
  if (type === 'node') {
    mapping = nodeColorsByDegree;
    assignedCounter = assignedNodeColorCount;
    defaultColor = DEFAULT_NODE_COLOR;
  } else { 
    mapping = edgeColorsByOverlap;
    assignedCounter = assignedEdgeColorCount;
    defaultColor = DEFAULT_EDGE_COLOR;
  }

  if (typeof value === 'undefined' || value === null) return defaultColor;

  if (mapping[value] === undefined) {
    if (assignedCounter < DISTINCT_COLORS.length) {
      mapping[value] = DISTINCT_COLORS[assignedCounter];
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

const nodeLegendItems = computed(() => Object.entries(nodeColorsByDegree).map(([value, color]) => ({ value: parseInt(value), color })).sort((a,b) => a.value - b.value));
const edgeLegendItems = computed(() => Object.entries(edgeColorsByOverlap).map(([value, color]) => ({ value: parseInt(value), color })).sort((a,b) => a.value - b.value));

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
const goToIMG2TXTPage = async () => { if (isProcessingSave.value) return; if (editMode.value && graphIsLoaded.value && modifications.value.length > 0) { isProcessingSave.value = true; try { await saveModifications(); router.push({ name: 'img-2-txt' }); } catch (err) { alert(`Error saving graph: ${err.message}. Cannot proceed to Annotate Text.`); } finally { isProcessingSave.value = false; } } else { router.push({ name: 'img-2-txt' }); }};
const saveGeneratedGraph = async (manuscriptName, page, graphData) => { try { const response = await fetch( import.meta.env.VITE_BACKEND_URL + `/save-graph/${manuscriptName}/${page}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ graph: graphData }) } ); if (!response.ok) throw new Error((await response.json()).error || 'Failed to save graph'); return await response.json(); } catch (error) { console.error('Error saving graph to backend:', error); return null; }};

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
  workingGraph.edges.forEach(edge => { edge.modified = edge.modified === undefined ? false : edge.modified; });
  resetSelection();
  resetColorMappingsAndPopulateLegends();
  console.log('Working graph reset.');
};

const resetSelection = () => { selectedNodes.value = []; tempLineEndPoint.value = null;};
const onNodeClick = (nodeId, event) => { if (isAKeyPressed.value || isDKeyPressed.value) return; event.stopPropagation(); const existingIndex = selectedNodes.value.indexOf(nodeId); if (existingIndex !== -1) { selectedNodes.value.splice(existingIndex, 1); } else { if (selectedNodes.value.length < 2) { selectedNodes.value.push(nodeId); } else { selectedNodes.value = [nodeId]; } } tempLineEndPoint.value = null;};
const onEdgeClick = (edge, event) => { if (isAKeyPressed.value || isDKeyPressed.value) return; event.stopPropagation(); selectedNodes.value = [edge.source, edge.target];};
const edgeExists = (nodeAId, nodeBId) => { if (nodeAId === undefined || nodeBId === undefined) return false; return workingGraph.edges.some(e => (e.source === nodeAId && e.target === nodeBId) || (e.source === nodeBId && e.target === nodeAId));};
const addEdgeManual = () => { if (selectedNodes.value.length !== 2) return; const [sourceId, targetId] = selectedNodes.value; if (sourceId === targetId || edgeExists(sourceId, targetId)) return; const newEdge = { source: sourceId, target: targetId, overlaps: 1, modified: true, label: 0 }; workingGraph.edges.push(newEdge); const sourceNode = getNodeById(sourceId); const targetNode = getNodeById(targetId); if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1; if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1; modifications.value.push({ type: 'add', ...newEdge }); resetColorMappingsAndPopulateLegends(); resetSelection();};
const addEdge = addEdgeManual; // Alias
const deleteEdgeManual = () => { if (selectedNodes.value.length !== 2) return; const [sourceId, targetId] = selectedNodes.value; const edgeIndex = workingGraph.edges.findIndex(e => (e.source === sourceId && e.target === targetId) || (e.source === targetId && e.target === sourceId)); if (edgeIndex === -1) return; const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0]; const sourceNode = getNodeById(removedEdge.source); const targetNode = getNodeById(removedEdge.target); if (sourceNode && sourceNode.numEdges > 0) sourceNode.numEdges--; if (targetNode && targetNode.numEdges > 0) targetNode.numEdges--; modifications.value.push({ type: 'delete', source: removedEdge.source, target: removedEdge.target, overlaps: removedEdge.overlaps, label: removedEdge.label !== undefined ? removedEdge.label : 0 }); resetColorMappingsAndPopulateLegends(); resetSelection();};
const deleteEdge = deleteEdgeManual; // Alias
const undoModification = (index) => { const mod = modifications.value[index]; if (mod.type === 'add') { const edgeIndex = workingGraph.edges.findIndex(e => e.source === mod.source && e.target === mod.target && e.modified); if (edgeIndex !== -1) { workingGraph.edges.splice(edgeIndex, 1); const sourceNode = getNodeById(mod.source); const targetNode = getNodeById(mod.target); if (sourceNode && sourceNode.numEdges > 0) sourceNode.numEdges--; if (targetNode && targetNode.numEdges > 0) targetNode.numEdges--; } } else if (mod.type === 'delete') { const reAddedEdge = { source: mod.source, target: mod.target, overlaps: mod.overlaps, label: mod.label, modified: true }; workingGraph.edges.push(reAddedEdge); const sourceNode = getNodeById(mod.source); const targetNode = getNodeById(mod.target); if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1; if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1; } modifications.value.splice(index, 1); resetColorMappingsAndPopulateLegends();};
const resetModifications = () => { resetWorkingGraph(); modifications.value = [];};
const isNodeSelected = (nodeId) => selectedNodes.value.includes(nodeId);
const isEdgeSelected = (edge) => { return selectedNodes.value.length === 2 && ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) || (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));};
const confirmAndNavigate = async (navigationAction) => { if (isProcessingSave.value) { alert("Please wait for the current save operation to complete."); return; } if (modifications.value.length > 0) { if (confirm('You have unsaved changes. Do you want to save them before navigating?')) { isProcessingSave.value = true; try { await saveModifications(); modifications.value = []; navigationAction(); } catch (err) { alert("Failed to save changes. Please try again or discard changes to navigate."); } finally { isProcessingSave.value = false; } } else { modifications.value = []; navigationAction(); } } else { navigationAction(); }};
const nextPage = () => confirmAndNavigate(() => annotationStore.nextPage());
const previousPage = () => confirmAndNavigate(() => annotationStore.previousPage());
const handleGlobalKeyDown = (e) => { if (e.key.toLowerCase() === 'e' && !e.ctrlKey && !e.metaKey) { if (isProcessingSave.value) return; e.preventDefault(); editMode.value = !editMode.value; return; } if (e.key.toLowerCase() === 't' && !e.ctrlKey && !e.metaKey) { if (loading.value || isProcessingSave.value) return; e.preventDefault(); goToIMG2TXTPage(); return; } if (!editMode.value || e.repeat) return; if (e.key.toLowerCase() === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); } if (e.key.toLowerCase() === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); }};
const handleGlobalKeyUp = (e) => { if (!editMode.value) return; if (e.key.toLowerCase() === 'd') isDKeyPressed.value = false; if (e.key.toLowerCase() === 'a') { isAKeyPressed.value = false; if (hoveredNodesForMST.size >= 2) addMSTEdges(); hoveredNodesForMST.clear(); }};
function distanceToLineSegment(px, py, x1, y1, x2, y2) { const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2; if (l2 === 0) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2); let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2; t = Math.max(0, Math.min(1, t)); const projX = x1 + t * (x2 - x1); const projY = y1 + t * (y2 - y1); return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);};
const handleEdgeHoverDelete = (mouseX_graph, mouseY_graph) => { const effectiveThreshold_graph = EDGE_HOVER_THRESHOLD_PX / currentScale.value; let edgeRemoved = false; for (let i = workingGraph.edges.length - 1; i >= 0; i--) { const edge = workingGraph.edges[i]; const nodeSource = getNodeById(edge.source); const nodeTarget = getNodeById(edge.target); if (!nodeSource || !nodeTarget) continue; const dist = distanceToLineSegment(mouseX_graph, mouseY_graph, nodeSource.x, nodeSource.y, nodeTarget.x, nodeTarget.y); if (dist < effectiveThreshold_graph) { const removedEdge = workingGraph.edges.splice(i, 1)[0]; if (nodeSource && nodeSource.numEdges > 0) nodeSource.numEdges--; if (nodeTarget && nodeTarget.numEdges > 0) nodeTarget.numEdges--; modifications.value.push({ type: 'delete', source: removedEdge.source, target: removedEdge.target, overlaps: removedEdge.overlaps, label: removedEdge.label !== undefined ? removedEdge.label : 0 }); edgeRemoved = true; } } if (edgeRemoved) resetColorMappingsAndPopulateLegends(); };
const handleNodeHoverCollect = (mouseX_graph, mouseY_graph) => { const actualHoverRadius_pixels = NODE_HOVER_RADIUS_PX / 2; const effectiveHoverRadius_graph = actualHoverRadius_pixels / currentScale.value; workingGraph.nodes.forEach(node => { const distSq = (mouseX_graph - node.x) ** 2 + (mouseY_graph - node.y) ** 2; if (distSq < effectiveHoverRadius_graph ** 2) { hoveredNodesForMST.add(node.id); } }); };
class DSU { constructor() { this.parent = {}; } init(nodeIndices) { this.parent = {}; nodeIndices.forEach(idx => this.parent[idx] = idx); } find(i) { if (this.parent[i] === i) return i; return this.parent[i] = this.find(this.parent[i]); } union(i, j) { const rootI = this.find(i); const rootJ = this.find(j); if (rootI !== rootJ) { this.parent[rootJ] = rootI; return true; } return false; }};
function calculateMST(nodeIds, allNodesData) { if (nodeIds.length < 2) return []; const nodesForMST = nodeIds.map(id => allNodesData.find(n => n.id === id)).filter(n => n); if (nodesForMST.length < 2) return []; const mstEdges = []; const potentialEdges = []; for (let i = 0; i < nodesForMST.length; i++) { for (let j = i + 1; j < nodesForMST.length; j++) { const p1 = nodesForMST[i]; const p2 = nodesForMST[j]; const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2); potentialEdges.push({ source: p1.id, target: p2.id, weight: dist }); } } potentialEdges.sort((a, b) => a.weight - b.weight); const dsu = new DSU(); dsu.init(nodeIds); for (const edge of potentialEdges) { if (dsu.union(edge.source, edge.target)) { mstEdges.push({ source: edge.source, target: edge.target }); } } return mstEdges;};
const addMSTEdges = () => { const nodesToConnect = Array.from(hoveredNodesForMST); if (nodesToConnect.length < 2) return; const mstNewEdges = calculateMST(nodesToConnect, workingGraph.nodes); let edgeAdded = false; mstNewEdges.forEach(edge => { if (!edgeExists(edge.source, edge.target)) { const newEdgeData = { source: edge.source, target: edge.target, overlaps: 1, modified: true, label: 0 }; workingGraph.edges.push(newEdgeData); const sourceNode = getNodeById(edge.source); const targetNode = getNodeById(edge.target); if (sourceNode) sourceNode.numEdges = (sourceNode.numEdges || 0) + 1; if (targetNode) targetNode.numEdges = (targetNode.numEdges || 0) + 1; modifications.value.push({ type: 'add', ...newEdgeData }); edgeAdded = true; } }); if (edgeAdded) resetColorMappingsAndPopulateLegends();};

// Save logic (unchanged)
const saveModifications = async () => { try { console.log('Saving modifications...'); const request = { graph: workingGraph, modifications: modifications.value, points: points.value.map(point => point.segment), modelName: annotationStore.modelName }; const response = await fetch( `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) } ); if (!response.ok) { const errorData = await response.json().catch(() => ({ error: "Failed to parse error from backend" })); throw new Error(errorData.error || 'Failed to save and process on backend'); } const responseData = await response.json(); graph.value = JSON.parse(JSON.stringify(workingGraph)); workingGraph.edges.forEach(edge => edge.modified = false); modifications.value = []; error.value = null; if (responseData.lines) { if (!annotationStore.recognitions[manuscriptName.value]) { annotationStore.recognitions[manuscriptName.value] = {}; } annotationStore.recognitions[manuscriptName.value][currentPage.value] = responseData.lines; } resetColorMappingsAndPopulateLegends(); console.log('Graph saved and page processed successfully.'); } catch (err) { console.error('Error in saveModifications:', err); error.value = err.message || 'Failed to save modifications'; throw err; }};
const saveModificationsAndStay = async () => { if (isProcessingSave.value) return; isProcessingSave.value = true; try { await saveModifications(); alert("Graph saved successfully!"); } catch (err) { alert(`Failed to save graph: ${err.message}`); } finally { isProcessingSave.value = false; }};

// --- IMAGE SAVING LOGIC (NEW) ---

/**
 * A helper function to load an image from a source URL using a Promise.
 * This allows us to use async/await for cleaner code.
 * @param {string} src - The source of the image (e.g., a base64 data URL).
 * @returns {Promise<HTMLImageElement>} A promise that resolves with the loaded image element.
 */
const loadImage = (src) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    // Allow cross-origin loading for SVGs, important for tainted canvas issues.
    img.crossOrigin = 'Anonymous'; 
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = src;
  });
};

/**
 * Generates and downloads a PNG image of the manuscript page with the graph overlay.
 * The image is rendered at the original resolution.
 */
const saveImageWithOverlay = async () => {
  if (!imageLoaded.value || !imageData.value || imageOriginalWidth.value === 0) {
    alert("Image data is not available to save.");
    return;
  }
  
  console.log("Starting image save process...");

  try {
    // 1. Create an off-screen canvas at original resolution
    const canvas = document.createElement('canvas');
    canvas.width = imageOriginalWidth.value;
    canvas.height = imageOriginalHeight.value;
    const ctx = canvas.getContext('2d');
    
    // Set a background color in case the image has transparency
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 2. Draw the base manuscript image onto the canvas
    const baseImageSrc = `data:image/jpeg;base64,${imageData.value}`;
    const baseImage = await loadImage(baseImageSrc);
    ctx.drawImage(baseImage, 0, 0, imageOriginalWidth.value, imageOriginalHeight.value);
    console.log("Base image drawn on canvas.");

    // 3. Generate an SVG string with non-scaled, original-resolution elements
    let svgString = `<svg width="${imageOriginalWidth.value}" height="${imageOriginalHeight.value}" xmlns="http://www.w3.org/2000/svg">`;

    // Draw edges first
    workingGraph.edges.forEach(edge => {
      const sourceNode = getNodeById(edge.source);
      const targetNode = getNodeById(edge.target);
      if (sourceNode && targetNode) {
        // For the export, we use the base stroke width, not the dynamically scaled one.
        const strokeWidth = BASE_EDGE_STROKE_WIDTH_SVG; 
        const strokeColor = getEdgeColor(edge);
        svgString += `<line x1="${sourceNode.x}" y1="${sourceNode.y}" x2="${targetNode.x}" y2="${targetNode.y}" stroke="${strokeColor}" stroke-width="${strokeWidth}" />`;
      }
    });

    // Draw nodes on top of edges
    workingGraph.nodes.forEach(node => {
      // For the export, we use the base radius, ignoring selection/hover states.
      const radius = BASE_NODE_RADIUS_SVG;
      // We also get the base color, ignoring interactive states.
      const nodeColor = getColorForValue(node.numEdges, 'node');
      svgString += `<circle cx="${node.x}" cy="${node.y}" r="${radius}" fill="${nodeColor}" />`;
    });

    svgString += `</svg>`;
    console.log("SVG string generated for overlay.");

    // 4. Draw the SVG overlay onto the canvas
    // We must first load the SVG string as an image
    const svgImageSrc = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`;
    const svgImage = await loadImage(svgImageSrc);
    ctx.drawImage(svgImage, 0, 0);
    console.log("SVG overlay drawn on canvas.");

    // 5. Trigger the download
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png'); // Use PNG for high quality
    link.download = `manuscript_${manuscriptName.value}_page_${currentPage.value}_with_graph.png`;
    
    // Append to body, click, and then remove
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log("Image download triggered successfully.");
    
  } catch (error) {
    console.error("Failed to save image with overlay:", error);
    alert("An error occurred while trying to save the image. Please check the console for details.");
  }
};


// --- LIFECYCLE HOOKS & WATCHERS ---
watch(() => annotationStore.currentPage, (newPage, oldPage) => {
  if (isProcessingSave.value) return;
  if (newPage && newPage !== oldPage) {
    console.log(`Current page changed to ${newPage}. Fetching data.`);
    fetchPageData();
  } else if (!newPage && oldPage) { // Navigated away from any page
    points.value = [];
    graph.value = { nodes: [], edges: [] };
    modifications.value = [];
    resetWorkingGraph();
    loading.value = false;
    error.value = null;
    imageData.value = '';
    imageLoaded.value = false;
    imageOriginalWidth.value = 0;
    imageOriginalHeight.value = 0;
  }
}, { immediate: true });

watch(editMode, (newValue) => {
  if (!newValue) { // Exiting edit mode
    resetSelection();
    isAKeyPressed.value = false;
    isDKeyPressed.value = false;
    hoveredNodesForMST.clear();
    tempLineEndPoint.value = null;
  }
});

watch([imageOriginalWidth, imageOriginalHeight], ([newW, newH]) => {
    if (newW > 0 && newH > 0 && visualizationContainerRef.value && imageLoaded.value) {
        console.log("Image dimensions watcher triggered, resetting view.");
        nextTick(resetView);
    }
});

let resizeObserver = null;
onMounted(() => {
  console.log('Component mounted.');
  window.addEventListener('keydown', handleGlobalKeyDown);
  window.addEventListener('keyup', handleGlobalKeyUp);

  nextTick(() => {
    if (visualizationContainerRef.value) {
      visualizationContainerRef.value.style.cursor = 'grab';
      if (imageLoaded.value && imageOriginalWidth.value > 0) {
          console.log("onMounted: image already loaded, resetting view.");
          resetView();
      }
      resizeObserver = new ResizeObserver(() => {
        console.log('Visualization container resized, resetting view.');
        resetView();
      });
      resizeObserver.observe(visualizationContainerRef.value);
    } else {
      console.warn('Visualization container ref not available on mount.');
    }
  });
});

onBeforeUnmount(() => {
  console.log('Component unmounting.');
  window.removeEventListener('keydown', handleGlobalKeyDown);
  window.removeEventListener('keyup', handleGlobalKeyUp);
  document.removeEventListener('mousemove', handlePanMove);
  document.removeEventListener('mouseup', handlePanEnd);
  if (resizeObserver && visualizationContainerRef.value) {
    resizeObserver.unobserve(visualizationContainerRef.value);
  }
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
});

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

.toolbar { display: flex; justify-content: space-between; align-items: flex-start;  padding: 3px 8px;  background-color: #f0f0f0; border-bottom: 1px solid #ddd; flex-shrink: 0; gap: 10px; }
.main-controls { display: flex; align-items: center; gap: 8px;  flex-wrap: nowrap;  }
.main-controls h10 { font-size: 0.9em;  white-space: nowrap; }
.main-controls button { padding: 3px 8px;  font-size: 0.85em; }
.toggle-container label { font-size: 0.85em; }
.legend-area { margin-left: auto;  }
.legend-container { display: flex; flex-direction: row;  gap: 8px; padding: 3px; font-size: 0.75em;  background-color: #f9f9f9; border-radius: 3px; border: 1px solid #e0e0e0; }
.legend h4 { margin-top: 0; margin-bottom: 3px; font-size: 0.9em;  font-weight: bold; }
.legend ul { list-style-type: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 3px; }
.legend li { display: flex; align-items: center; gap: 3px; padding: 1px 3px; border: 1px solid #eee; border-radius: 2px; background-color: #fff; }
.color-box { width: 10px; height: 10px; border: 1px solid #ccc; display: inline-block; }
.legend-modified-edge-note { margin-top: 2px; display: flex; align-items: center; gap: 3px; font-style: italic; font-size: 0.9em; }
.edit-instructions-bar { background-color: #e9ecef;  padding: 3px 8px; font-size: 0.8em; color: #495057; border-bottom: 1px solid #ddd; text-align: center; flex-shrink: 0; }
.edit-instructions-bar p { margin: 0; display: inline;  margin-right: 10px;  }
.processing-save-notice, .loading, .error-message {  padding: 15px; text-align: center; flex-shrink: 0; }
.processing-save-notice { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: rgba(0,0,0,0.75); color: white; border-radius: 8px; z-index: 10000; font-size: 1em; }
.loading { font-style: italic; color: #666; flex-grow: 1; display:flex; align-items:center; justify-content:center;}
.error-message { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; margin: 5px; border-radius: 4px; }

/* --- Styles for Zoom/Pan --- */
.visualization-container {
  position: relative;
  overflow: hidden;
  flex-grow: 1; 
  background-color: #e0e0e0; 
  display: flex;
  cursor: grab;
  user-select: none;
}

.transform-wrapper {
  position: relative; 
}

.image-wrapper {
  position: relative;
  line-height: 0;
}

.manuscript-image {
  display: block; 
  max-width: none;
  max-height: none;
  user-select: none;
  pointer-events: none;
  opacity: 0.85; 
}

.placeholder-image {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #ccc;
  color: #666;
}

.points-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.point {
  position: absolute;
  background-color: rgba(255, 0, 0, 0.7);
  border-radius: 50%;
}

.graph-overlay {
  position: absolute;
  top: 0;
  left: 0;
  cursor: default;
  pointer-events: auto;
}
/* --- End Styles for Zoom/Pan --- */

.modifications-log-container { padding: 5px 8px; background-color: #f0f0f0; border-top: 1px solid #ddd; flex-shrink: 0; display: flex;  align-items: center; gap: 10px; font-size: 0.8em; }
.modifications-log-container > button {  padding: 3px 8px; font-size: 0.9em;  }
.modifications-details { display: flex; align-items: center; gap: 5px; flex-grow: 1;  }
.modifications-details span {  white-space: nowrap; }
.modifications-details > button {  padding: 2px 6px; font-size: 0.9em; margin-left: 5px;  }
.modifications-details ul { list-style-type: none; padding: 0 3px; margin: 0; max-height: 40px;  overflow-y: auto; border: 1px solid #ddd; background-color: #fff; border-radius: 3px; flex-grow: 1;  display: flex;  flex-direction: column; }
.modification-item { display: flex; justify-content: space-between; align-items: center; padding: 2px 4px; border-bottom: 1px solid #eee; font-size: 0.9em;  white-space: nowrap; }
.modification-item:last-child { border-bottom: none; }
.undo-button { font-size: 0.9em; padding: 1px 4px; background-color: #fffde7; border-color: #fff59d; margin-left: 5px; }
button:disabled {  opacity: 0.5; cursor: not-allowed; background-color: #e9ecef; }
button:hover:not(:disabled) { background-color: #dde1e6; }
</style>