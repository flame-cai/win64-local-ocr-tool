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
            :fill="isNodeSelected(index) ? '#ff9500' : '#2ecc71'"
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
import { ref, onMounted, onUnmounted, computed, watch, reactive } from 'vue';
import { useAnnotationStore } from '@/stores/annotationStore';

const annotationStore = useAnnotationStore();

const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
const currentPage = computed(() => annotationStore.currentPage);

const loading = ref(true);
const error = ref(null);
const dimensions = ref([0, 0]);
const points = ref([]);
const graph = ref({ nodes: [], edges: [] });
const imageData = ref('');
const imageLoaded = ref(false);
const showPoints = ref(true);
const showGraph = ref(true);

// Editing state
const editMode = ref(false);
const selectedNodes = ref([]);
const tempEndPoint = ref(null);
const modifications = ref([]);
const workingGraph = reactive({ nodes: [], edges: [] });

// Scale factor (similar to the Python code's resize)
const scaleFactor = 0.5; // This is equivalent to dividing by 2 as in your Python code

// Calculate scaled dimensions
const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor));
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor));

// Scale functions to map original coordinates to scaled view
const scaleX = (x) => x * scaleFactor;
const scaleY = (y) => y * scaleFactor;

// Container ref for potential scrolling/zooming features
const container = ref(null);

const updateCanvasSize = (width, height) => {
  dimensions.value = [width, height];
};

const fetchPageData = async () => {
  if (!manuscriptName.value || !currentPage.value) return;
  
  loading.value = true;
  error.value = null;
  points.value = [];
  graph.value = { nodes: [], edges: [] };
  imageData.value = '';
  imageLoaded.value = false;
  
  try {
    console.log(`Fetching data for manuscript: ${manuscriptName.value}, page: ${currentPage.value}`);
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + `/semi-segment/${manuscriptName.value}/${currentPage.value}`
    );
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch page data');
    }
    
    const data = await response.json();
    console.log("Received data:", Object.keys(data));
    
    // Update canvas size
    updateCanvasSize(data.dimensions[0], data.dimensions[1]);
    
    // Process points
    points.value = data.points.map(point => ({
      coordinates: [point[0], point[1]],
      segment: null,
    }));
    
    // Process graph
    if (data.graph) {
      graph.value = data.graph;
      // Clone to working graph
      resetWorkingGraph();
    }
    
    // Process image
    if (data.image) {
      console.log(`Loading image data, length: ${data.image.length}`);
      imageData.value = data.image;
    } else {
      console.log("No image data found in response");
    }
  } catch (err) {
    console.error('Error fetching page data:', err);
    error.value = err.message || 'Failed to load page data';
  } finally {
    loading.value = false;
  }
};

const resetWorkingGraph = () => {
  // Deep clone the original graph to working graph
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
  resetSelection();
  modifications.value = [];
};

const resetSelection = () => {
  selectedNodes.value = [];
  tempEndPoint.value = null;
};

const onNodeClick = (nodeIndex, event) => {
  event.stopPropagation();
  
  // If node is already selected, deselect it
  const existingIndex = selectedNodes.value.indexOf(nodeIndex);
  if (existingIndex !== -1) {
    selectedNodes.value.splice(existingIndex, 1);
    return;
  }
  
  // Add to selection (but limit to 2 nodes)
  if (selectedNodes.value.length < 2) {
    selectedNodes.value.push(nodeIndex);
  } else {
    // Replace selection if already have 2 nodes
    selectedNodes.value = [nodeIndex];
  }
  
  tempEndPoint.value = null;
};

const onEdgeClick = (edge, event) => {
  event.stopPropagation();
  
  // Select the nodes that form this edge
  selectedNodes.value = [edge.source, edge.target];
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
  
  // Check if edge already exists
  if (edgeExists(source, target)) {
    console.log('Edge already exists');
    return;
  }
  
  // Add edge to working graph
  workingGraph.edges.push({
    source,
    target,
    label: 0, // Default label for same-line connection
    modified: true
  });
  
  // Track modification
  modifications.value.push({
    type: 'add',
    source,
    target,
    label: 0
  });
  
  resetSelection();
};

const deleteEdge = () => {
  if (selectedNodes.value.length !== 2) return;
  
  const [source, target] = selectedNodes.value;
  
  // Find the edge index
  const edgeIndex = workingGraph.edges.findIndex(e => 
    (e.source === source && e.target === target) ||
    (e.source === target && e.target === source)
  );
  
  if (edgeIndex === -1) {
    console.log('Edge not found');
    return;
  }
  
  // Track modification before removing
  const removedEdge = workingGraph.edges[edgeIndex];
  modifications.value.push({
    type: 'delete',
    source: removedEdge.source,
    target: removedEdge.target,
    label: removedEdge.label
  });
  
  // Remove edge
  workingGraph.edges.splice(edgeIndex, 1);
  
  resetSelection();
};

const undoModification = (index) => {
  const mod = modifications.value[index];
  
  if (mod.type === 'add') {
    // Find and remove the added edge
    const edgeIndex = workingGraph.edges.findIndex(e => 
      (e.source === mod.source && e.target === mod.target) ||
      (e.source === mod.target && e.target === mod.source)
    );
    
    if (edgeIndex !== -1) {
      workingGraph.edges.splice(edgeIndex, 1);
    }
  } else if (mod.type === 'delete') {
    // Re-add the deleted edge
    workingGraph.edges.push({
      source: mod.source,
      target: mod.target,
      label: mod.label
    });
  }
  
  // Remove this modification from the list
  modifications.value.splice(index, 1);
};

const resetModifications = () => {
  resetWorkingGraph();
};

// #TODO add code to save the updated graph
const isNodeSelected = (nodeIndex) => {
  return selectedNodes.value.includes(nodeIndex);
};

const isEdgeSelected = (edge) => {
  return selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
     (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));
};

const getEdgeColor = (edge) => {
  // Modified edges get a different color
  if (edge.modified) return '#ff9500';
  // Original edge coloring logic
  return edge.label === 0 ? '#3498db' : '#e74c3c';
};

const nextPage = () => {
  // Check for unsaved changes
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before moving to the next page?')) {
      saveModifications();
    }
  }
  annotationStore.setCurrentPage(String(Number(currentPage.value) + 1));
};

const previousPage = () => {
  // Check for unsaved changes
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before moving to the previous page?')) {
      saveModifications();
    }
  }
  if (Number(currentPage.value) > 1) {
    annotationStore.setCurrentPage(String(Number(currentPage.value) - 1));
  }
};

// Add mouse move handler for visualization when selecting nodes
const handleMouseMove = (event) => {
  if (!editMode.value || selectedNodes.value.length !== 1) return;
  
  const rect = container.value.getBoundingClientRect();
  tempEndPoint.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
};

// Watch for page changes
watch(
  () => annotationStore.currentPage,
  () => {
    // Check for unsaved changes
    if (modifications.value.length > 0) {
      if (confirm('You have unsaved changes. Do you want to save them before changing pages?')) {
        saveModifications();
      } else {
        modifications.value = [];
      }
    }
    fetchPageData();
  }
);

// Watch for edit mode toggle
watch(editMode, (newValue) => {
  if (newValue) {
    // Add mouse move listener when entering edit mode
    document.addEventListener('mousemove', handleMouseMove);
  } else {
    // Remove listener when leaving edit mode
    document.removeEventListener('mousemove', handleMouseMove);
    resetSelection();
  }
});

// Initial data fetch
onMounted(() => {
  fetchPageData();
});

// Clean up
onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove);
});

const saveModifications = async () => {
  try {
    console.log('Saving modifications and generating line labels...');
    
    // Prepare the request with the modified graph data
    const request = {
      graph: workingGraph,
      modifications: modifications.value,
      points: points.value.map(point => point.segment)
    };
    
    const response = await fetch(
      import.meta.env.VITE_BACKEND_URL + 
      `/semi-segment/${manuscriptName.value}/${currentPage.value}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
    
    if (!response.ok) {
      throw new Error('Failed to save modifications and generate labels');
    }
    
    // Update the original graph with the working graph
    graph.value = JSON.parse(JSON.stringify(workingGraph));
    modifications.value = [];
    
    console.log('Graph modifications saved and labels generated successfully');
  } catch (err) {
    console.error('Error saving modifications:', err);
    error.value = err.message || 'Failed to save modifications';
  }
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