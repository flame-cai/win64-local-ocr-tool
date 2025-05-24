<template>
  <div class="manuscript-viewer">
    <div class="toolbar">
      <h2>{{ manuscriptName }} - Page {{ currentPage }}</h2>
      <div class="controls">
        <button @click="previousPage" :disabled="loading">Previous</button>
        <button @click="nextPage" :disabled="loading">Next</button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="showPoints" />
            Show Points
          </label>
          <label>
            <input type="checkbox" v-model="showGraph" />
            Show Graph
          </label>
          <label>
            <input type="checkbox" v-model="editMode" />
            Edit Mode
          </label>
        </div>
      </div>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="loading" class="loading">
      Loading page data...
    </div>

    <div v-else class="visualization-container" ref="container">
      <div class="image-container" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
        <!-- Background image -->
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
        
        <!-- Points overlay -->
        <div 
          v-if="showPoints && points.length > 0" 
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
        
        <!-- Graph overlay -->
        <svg 
          v-if="showGraph && workingGraph.nodes && workingGraph.edges" 
          class="graph-overlay"
          :width="scaledWidth"
          :height="scaledHeight"
          @click="editMode && onBackgroundClick"
        >
          <!-- Edges -->
          <line
            v-for="(edge, index) in workingGraph.edges"
            :key="`edge-${index}`"
            :x1="scaleX(workingGraph.nodes[edge.source].x)"
            :y1="scaleY(workingGraph.nodes[edge.source].y)"
            :x2="scaleX(workingGraph.nodes[edge.target].x)"
            :y2="scaleY(workingGraph.nodes[edge.target].y)"
            :stroke="getEdgeColor(edge)"
            :stroke-width="isEdgeSelected(edge) ? 3 : 1.5"
            :stroke-opacity="0.7"
            @click="editMode && onEdgeClick(edge, $event)"
          />
          
          <!-- Nodes -->
          <circle
            v-for="(node, index) in workingGraph.nodes"
            :key="`node-${index}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="isNodeSelected(index) ? 6 : 3"
            :fill="getNodeFill(index)"
            :fill-opacity="0.8"
            @click="editMode && onNodeClick(index, $event)"
          />
          
          <!-- Selection line (when one node is selected) -->
          <line
            v-if="editMode && selectedNodes.length === 1 && tempEndPoint"
            :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
            :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
            :x2="tempEndPoint.x"
            :y2="tempEndPoint.y"
            stroke="#ff9500"
            stroke-width="1.5"
            stroke-dasharray="5,5"
            stroke-opacity="0.7"
          />
        </svg>
      </div>
    </div>

    <div v-if="editMode" class="edit-controls">
      <div class="edit-instructions">
        <p v-if="selectedNodes.length === 0">Select first node to create/delete edge</p>
        <p v-else-if="selectedNodes.length === 1">Select second node to create/delete edge</p>
        <p v-else>Click "Add Edge" or "Delete Edge" below</p>
      </div>
      <div class="edit-actions">
        <button @click="resetSelection">Cancel</button>
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

    <div v-if="modifications.length > 0" class="modifications-log">
        <h3>Modifications ({{ modifications.length }})</h3>
        <button @click="saveModifications">Save Changes</button>
        <button @click="resetModifications">Reset All</button>
        <ul>
          <li v-for="(mod, index) in modifications" :key="index" class="modification-item">
            {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge between Node {{ mod.source }} and Node {{ mod.target }}
            <button @click="undoModification(index)" class="undo-button">Undo</button>
          </li>
        </ul>
      </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, onUnmounted, computed, watch, reactive } from 'vue';
import { useAnnotationStore } from '@/stores/annotationStore';

const handleKeydown = (e) => {
  if (!editMode.value) return;
  if (e.key === 'a') addEdge();
  if (e.key === 'd') deleteEdge();
};

onMounted(() => window.addEventListener('keydown', handleKeydown));
onBeforeUnmount(() => window.removeEventListener('keydown', handleKeydown));

const annotationStore = useAnnotationStore();

const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');

// --- Local Page Management ---
// Initialize localCurrentPage. Try from store if available, fallback to '1'.
// Page numbers are kept as strings for consistency.
const localCurrentPage = ref(String(annotationStore.currentPage || '1'));
// currentPage computed property now uses the local state.
const currentPage = computed(() => localCurrentPage.value);

// Internal function to update localCurrentPage.
// This will trigger the watcher to fetch new page data.
const _setCurrentPageInternal = (newPage) => {
  const pageStr = String(newPage);
  if (localCurrentPage.value !== pageStr) {
    localCurrentPage.value = pageStr;
  }
};
// --- End Local Page Management ---

const loading = ref(true);
const error = ref(null);
const dimensions = ref([0, 0]);
const points = ref([]); // Raw points from data.points
const graph = ref({ nodes: [], edges: [] }); // Original graph from server for the current page
const imageData = ref('');
const imageLoaded = ref(false);
const showPoints = ref(true);
const showGraph = ref(true);

// Editing state
const editMode = ref(true);
const selectedNodes = ref([]);
const tempEndPoint = ref(null);
const modifications = ref([]);
const workingGraph = reactive({ nodes: [], edges: [] }); // Editable copy of the graph

// Scale factor
const scaleFactor = 0.5;
const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor));
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor));

const scaleX = (x) => x * scaleFactor;
const scaleY = (y) => y * scaleFactor;

const container = ref(null); // Ref for the SVG container

const updateCanvasSize = (width, height) => {
  dimensions.value = [width, height];
};

const fetchPageData = async () => {
  if (!manuscriptName.value || !currentPage.value) {
    console.log("Skipping fetch: manuscriptName or currentPage is not set.", manuscriptName.value, currentPage.value);
    loading.value = false; // Ensure loading state is reset
    return;
  }
  
  loading.value = true;
  error.value = null;
  
  // Reset states for the new page
  resetSelection();
  modifications.value = [];
  
  try {
    console.log(`Fetching data for manuscript: ${manuscriptName.value}, page: ${currentPage.value}`);
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`
    );
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || errorData.detail || 'Failed to fetch page data');
    }
    
    const data = await response.json();
    console.log("Received data for page:", currentPage.value, "Keys:", Object.keys(data));
    
    updateCanvasSize(data.dimensions[0], data.dimensions[1]);
    
    points.value = data.points.map(point => ({
      coordinates: [point[0], point[1]],
      segment: null,
    }));
    
    if (data.graph) {
      graph.value = data.graph;
    } else {
      graph.value = { nodes: [], edges: [] }; // Ensure graph is reset if not in data
    }
    resetWorkingGraph(); // Initialize workingGraph from the new graph.value
    
    if (data.image) {
      imageData.value = data.image;
      imageLoaded.value = true; // Assume loaded if data URL is present
    } else {
      imageData.value = '';
      imageLoaded.value = false;
    }
  } catch (err) {
    console.error('Error fetching page data:', err);
    error.value = err.message || 'Failed to load page data';
    // Clear data on error to avoid showing stale information
    points.value = [];
    graph.value = { nodes: [], edges: [] };
    resetWorkingGraph(); // Reset to empty graph
    imageData.value = '';
    imageLoaded.value = false;
  } finally {
    loading.value = false;
  }
};

const resetWorkingGraph = () => {
  // Deep clone the original graph for the current page to the working graph
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
  // Selections and modifications are typically reset when page changes or explicitly.
};

const resetSelection = () => {
  selectedNodes.value = [];
  tempEndPoint.value = null;
};

const onNodeClick = (nodeIndex, event) => {
  event.stopPropagation();
  const existingIndex = selectedNodes.value.indexOf(nodeIndex);
  if (existingIndex !== -1) {
    selectedNodes.value.splice(existingIndex, 1); // Deselect if already selected
  } else {
    if (selectedNodes.value.length < 2) {
      selectedNodes.value.push(nodeIndex); // Add to selection (max 2)
    } else {
      selectedNodes.value = [nodeIndex]; // Replace selection if 2 already selected
    }
  }
  tempEndPoint.value = null; // Clear temporary line on new click
};

const onEdgeClick = (edge, event) => {
  event.stopPropagation();
  selectedNodes.value = [edge.source, edge.target]; // Select the two nodes forming the edge
};

const onBackgroundClick = () => {
  resetSelection();
};

const edgeExists = (nodeA, nodeB) => {
  return workingGraph.edges.some(e => 
    (e.source === nodeA && e.target === nodeB) || 
    (e.source === nodeB && e.target === nodeA)
  );
};

const addEdge = () => {
  if (selectedNodes.value.length !== 2) return;
  const [source, target] = selectedNodes.value;
  if (edgeExists(source, target)) {
    console.log('Edge already exists');
    return;
  }
  const newEdge = { source, target, label: 0, modified: true }; // Default label, mark as modified
  workingGraph.edges.push(newEdge);
  modifications.value.push({ type: 'add', source, target, label: 0 }); // Track modification
  resetSelection();
};

const deleteEdge = () => {
  if (selectedNodes.value.length !== 2) return;
  const [source, target] = selectedNodes.value;
  const edgeIndex = workingGraph.edges.findIndex(e => 
    (e.source === source && e.target === target) ||
    (e.source === target && e.target === source)
  );
  if (edgeIndex === -1) {
    console.log('Edge not found');
    return;
  }
  const removedEdge = workingGraph.edges[edgeIndex];
  modifications.value.push({ // Track deletion
    type: 'delete',
    source: removedEdge.source,
    target: removedEdge.target,
    label: removedEdge.label 
  });
  workingGraph.edges.splice(edgeIndex, 1); // Remove edge
  resetSelection();
};

const undoModification = (index) => {
  const mod = modifications.value[index];
  if (mod.type === 'add') {
    // Find and remove the added edge (the one marked as 'modified' potentially)
    const edgeIndex = workingGraph.edges.findIndex(e => 
      e.source === mod.source && e.target === mod.target && e.modified
    );
    if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1);
  } else if (mod.type === 'delete') {
    // Re-add the deleted edge (not marked as 'modified' unless it was originally so)
    workingGraph.edges.push({ source: mod.source, target: mod.target, label: mod.label });
  }
  modifications.value.splice(index, 1); // Remove this modification from the list
};

const resetModifications = () => {
  resetWorkingGraph(); // Re-clones from original graph.value for the current page
  modifications.value = []; // Clear modification log
  resetSelection(); // Clear any active selections
};

const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex);

const isEdgeSelected = (edge) => {
  return selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
     (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));
};

const getEdgeColor = (edge) => {
  if (edge.modified) return '#ff9500'; // Orange for modified edges
  return edge.label === 0 ? '#3498db' : '#e74c3c'; // Blue for same-line, Red for cross-line (original logic)
};

// --- Page Navigation with Local setCurrentPage ---
const nextPage = async () => {
  if (modifications.value.length > 0) {
    const confirmSave = confirm('You have unsaved changes. Do you want to save them before moving to the next page?');
    if (confirmSave) {
      try {
        await saveModifications(); // saveModifications clears modifications on success
      } catch (e) {
        console.error("Save failed, not navigating:", e);
        return; // Save failed, do not navigate
      }
    } else {
      // User chose not to save, discard changes
      modifications.value = [];
      resetWorkingGraph(); // Revert working graph to original state for current page
    }
  }
  // Proceed to next page
  const nextPageNumber = Number(currentPage.value) + 1;
  // TODO: Add a check for max page number if available from store or props
  _setCurrentPageInternal(String(nextPageNumber));
};

const previousPage = async () => {
  if (modifications.value.length > 0) {
    const confirmSave = confirm('You have unsaved changes. Do you want to save them before moving to the previous page?');
    if (confirmSave) {
      try {
        await saveModifications();
      } catch (e) {
        console.error("Save failed, not navigating:", e);
        return; // Save failed, do not navigate
      }
    } else {
      modifications.value = [];
      resetWorkingGraph();
    }
  }
  // Proceed to previous page
  const prevPageNumber = Number(currentPage.value) - 1;
  if (prevPageNumber >= 1) { // Assuming pages are 1-indexed
    _setCurrentPageInternal(String(prevPageNumber));
  }
};
// --- End Page Navigation ---

const handleMouseMove = (event) => {
  if (!editMode.value || !container.value || selectedNodes.value.length !== 1) return;
  const rect = container.value.getBoundingClientRect();
  tempEndPoint.value = {
    x: event.clientX - rect.left, // Position relative to the container
    y: event.clientY - rect.top
  };
};

// Watch for local page changes to fetch data
watch(localCurrentPage, (newPage, oldPage) => {
  // Ensure this watcher doesn't fire for the initial value or no-change scenarios
  if (newPage && newPage !== oldPage) {
    console.log(`Local current page changed from ${oldPage} to ${newPage}. Fetching data.`);
    // Unsaved changes are handled *before* localCurrentPage is updated by nextPage/previousPage.
    // So, we can directly fetch data here.
    fetchPageData();
  }
});

watch(editMode, (newValue, oldValue) => {
  if (newValue === oldValue) return; // No change
  if (newValue) {
    // Add mouse move listener when entering edit mode (typically on document for wider capture area)
    document.addEventListener('mousemove', handleMouseMove);
  } else {
    // Remove listener when leaving edit mode
    document.removeEventListener('mousemove', handleMouseMove);
    resetSelection(); // Clear selection when exiting edit mode
  }
});

// Initial data fetch and setup
onMounted(() => {
  // localCurrentPage is already initialized.
  // fetchPageData will use the correct initial page.
  fetchPageData();
  
  // If edit mode is initially true, add the mouse move listener
  if (editMode.value) {
    document.addEventListener('mousemove', handleMouseMove);
  }
});

// Clean up listeners
onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove);
  // The keydown listener is handled by onBeforeUnmount, which is appropriate.
});

const saveModifications = async () => {
  try {
    console.log('Saving modifications for page:', currentPage.value);
    // Backend expects the full graph structure reflecting all changes.
    const payload = {
      graph: workingGraph,
      // If your backend also needs points or segment data, include it here:
      // points: points.value.map(point => point.segment), 
    };
    
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || errorData.error || 'Failed to save modifications');
    }
    
    // const result = await response.json(); // Process result if backend returns updated data

    // Update the base graph (graph.value) to reflect the saved changes
    graph.value = JSON.parse(JSON.stringify(workingGraph));
    // Clear pending modifications list as they are now saved
    modifications.value = [];
    
    // Mark all edges in the workingGraph as no longer 'modified' since they are now persisted
    workingGraph.edges.forEach(edge => {
      delete edge.modified;
    });
    
    console.log('Graph modifications saved successfully for page:', currentPage.value);
    // Optionally, you might want to re-fetch data if the backend process modifies it further (e.g., re-labels)
    // await fetchPageData(); // Uncomment if a full refresh is needed post-save

  } catch (err) {
    console.error('Error saving modifications:', err);
    error.value = err.message || 'Failed to save modifications';
    throw err; // Re-throw so calling functions (nextPage/previousPage) can handle it
  }
};

// --- Node Degree Calculation and Styling ---
const nodeDegrees = computed(() => {
  const degrees = {};
  if (!workingGraph.nodes || workingGraph.nodes.length === 0) {
    return degrees;
  }
  // Initialize degrees for all node indices based on workingGraph.nodes
  for (let i = 0; i < workingGraph.nodes.length; i++) {
    degrees[i] = 0;
  }
  // Calculate degrees by iterating through edges
  for (const edge of workingGraph.edges) {
    if (degrees[edge.source] !== undefined) {
      degrees[edge.source]++;
    }
    if (degrees[edge.target] !== undefined) {
      degrees[edge.target]++;
    }
  }
  return degrees;
});

// Function to determine if a node has a high degree (3 or more edges)
const isNodeHighDegree = (nodeIndex) => {
  // nodeDegrees.value might be {} if nodes array is empty
  return (nodeDegrees.value[nodeIndex] || 0) >= 3;
};

// Helper function intended for use in the template to determine node fill color.
// Example usage in template: <circle ... :fill="getNodeFill(nodeIndex)" />
// All top-level bindings in <script setup> are automatically available to the template.
const getNodeFill = (nodeIndex) => {
  if (isNodeHighDegree(nodeIndex)) {
    return 'red'; // Nodes with 3+ edges are red
  }
  if (isNodeSelected(nodeIndex)) {
    return '#ff9500'; // Selected nodes are orange (example color)
  }
  // Provide a default color for nodes if not high-degree or selected
  return '#888888'; // A generic gray, adjust as needed
};

</script>

<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
}

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.toggle-container {
  display: flex;
  gap: 8px;
}

.visualization-container {
  position: relative;
  overflow: auto;
  flex: 1;
  background-color: #eee;
}

.image-container {
  position: relative;
  margin: 0 auto;
}

.manuscript-image {
  display: block;
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
  pointer-events: none;
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
  width: 100%;
  height: 100%;
}

.loading {
  padding: 20px;
  text-align: center;
  font-style: italic;
  color: #666;
}

.error-message {
  padding: 20px;
  background-color: #fee;
  color: #c00;
  border: 1px solid #faa;
  margin: 10px;
  border-radius: 4px;
}

/* Edit mode styling */
.edit-controls {
  padding: 10px;
  background-color: #f9f9f9;
  border-bottom: 1px solid #ddd;
}

.edit-instructions {
  margin-bottom: 10px;
  font-size: 14px;
  color: #555;
}

.edit-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 15px;
}

button {
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid #ccc;
  background-color: #fff;
  cursor: pointer;
}

button:hover {
  background-color: #f0f0f0;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.modifications-log {
  border-top: 1px solid #ddd;
  padding-top: 10px;
  margin-top: 10px;
}

.modifications-log h3 {
  font-size: 16px;
  margin-bottom: 10px;
}

.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 0;
  border-bottom: 1px solid #eee;
}

.undo-button {
  font-size: 12px;
  padding: 2px 6px;
}
</style>