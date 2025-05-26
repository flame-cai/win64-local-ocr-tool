// layoutGraphGenerator.js
/**
 * Build a KD-Tree for fast neighbor lookup
 */
class KDTree {
  constructor(points) {
    this.points = points;
    this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  buildTree(points, depth) {
    if (points.length === 0) return null;
    if (points.length === 1) return points[0];

    const k = 2; // 2D points
    const axis = depth % k;
    
    points.sort((a, b) => a.point[axis] - b.point[axis]);
    const median = Math.floor(points.length / 2);
    
    return {
      point: points[median].point,
      index: points[median].index,
      left: this.buildTree(points.slice(0, median), depth + 1),
      right: this.buildTree(points.slice(median + 1), depth + 1),
      axis: axis
    };
  }

  query(queryPoint, k) {
    const best = [];
    
    const search = (node, depth) => {
      if (!node) return;
      
      const distance = this.euclideanDistance(queryPoint, node.point);
      
      if (best.length < k) {
        best.push({ distance, index: node.index });
        best.sort((a, b) => a.distance - b.distance);
      } else if (distance < best[best.length - 1].distance) {
        best[best.length - 1] = { distance, index: node.index };
        best.sort((a, b) => a.distance - b.distance);
      }
      
      const axis = depth % 2;
      const diff = queryPoint[axis] - node.point[axis];
      
      const closer = diff < 0 ? node.left : node.right;
      const farther = diff < 0 ? node.right : node.left;
      
      search(closer, depth + 1);
      
      if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
        search(farther, depth + 1);
      }
    };
    
    search(this.tree, 0);
    return best.map(b => b.index);
  }

  euclideanDistance(p1, p2) {
    return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
  }
}

/**
 * DBSCAN clustering implementation to identify majority cluster and outliers
 */
function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
  if (toCluster.length === 0) return [];
  
  // DBSCAN implementation
  const labels = dbscan(toCluster, eps, minSamples);
  
  // Count the occurrences of each label
  const labelCounts = {};
  labels.forEach(label => {
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  });
  
  // Find the majority cluster label (excluding -1 outliers)
  let majorityLabel = null;
  let maxCount = 0;
  
  for (const [label, count] of Object.entries(labelCounts)) {
    const labelNum = parseInt(label);
    if (labelNum !== -1 && count > maxCount) {
      majorityLabel = labelNum;
      maxCount = count;
    }
  }
  
  // Create a new label array where the majority cluster is 0 and all others are -1
  const newLabels = new Array(labels.length).fill(-1); // Initialize all as outliers
  
  if (majorityLabel !== null) {
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === majorityLabel) {
        newLabels[i] = 0; // Assign 0 to the majority cluster
      }
    }
  }
  
  return newLabels;
}

/**
 * DBSCAN clustering algorithm implementation
 */
function dbscan(points, eps, minSamples) {
  const labels = new Array(points.length).fill(-1); // -1 means unclassified
  let clusterId = 0;
  
  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue; // Already processed
    
    const neighbors = getNeighbors(points, i, eps);
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Mark as noise/outlier
    } else {
      // Start a new cluster
      expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
      clusterId++;
    }
  }
  
  return labels;
}

/**
 * Get neighbors within eps distance
 */
function getNeighbors(points, pointIndex, eps) {
  const neighbors = [];
  const point = points[pointIndex];
  
  for (let i = 0; i < points.length; i++) {
    if (euclideanDistance(point, points[i]) <= eps) {
      neighbors.push(i);
    }
  }
  
  return neighbors;
}

/**
 * Expand cluster by adding density-reachable points
 */
function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
  labels[pointIndex] = clusterId;
  
  let i = 0;
  while (i < neighbors.length) {
    const neighborIndex = neighbors[i];
    
    if (labels[neighborIndex] === -1) {
      labels[neighborIndex] = clusterId;
      
      const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
      if (neighborNeighbors.length >= minSamples) {
        // Add new neighbors to the list (union operation)
        for (const newNeighbor of neighborNeighbors) {
          if (!neighbors.includes(newNeighbor)) {
            neighbors.push(newNeighbor);
          }
        }
      }
    }
    
    i++;
  }
}

function euclideanDistance(p1, p2) {
  return Math.sqrt(p1.reduce((sum, val, i) => sum + (val - p2[i]) ** 2, 0));
}

/**
 * Generate a graph representation of text layout based on points.
 * This function implements the core layout analysis logic.
 */
export function generateLayoutGraph(points) {
  const NUM_NEIGHBOURS = 6;
  const cos_similarity_less_than = -0.8;
  
  // Build a KD-tree for fast neighbor lookup
  const tree = new KDTree(points);
  const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
  
  // Store graph edges and their properties
  const edges = [];
  const edgeProperties = [];
  
  // Process nearest neighbors
  for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
    const nbrIndices = indices[currentPointIndex];
    const currentPoint = points[currentPointIndex];
    
    const normalizedPoints = nbrIndices.map(idx => [
      points[idx][0] - currentPoint[0],
      points[idx][1] - currentPoint[1]
    ]);
    
    const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
    const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
    
    // Create a list of relative neighbors with their global indices
    const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
      globalIdx,
      scaledPoint: scaledPoints[i],
      normalizedPoint: normalizedPoints[i]
    }));
    
    const filteredNeighbours = [];
    
    for (let i = 0; i < relativeNeighbours.length; i++) {
      for (let j = i + 1; j < relativeNeighbours.length; j++) {
        const neighbor1 = relativeNeighbours[i];
        const neighbor2 = relativeNeighbours[j];
        
        const norm1 = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
        const norm2 = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
        
        let cosSimilarity = 0.0;
        if (norm1 * norm2 !== 0) {
          const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                           neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
          cosSimilarity = dotProduct / (norm1 * norm2);
        }
        
        // Calculate non-normalized distances
        const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
        const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
        const totalLength = norm1Real + norm2Real;
        
        // Select pairs with angles close to 180 degrees (opposite directions)
        if (cosSimilarity < cos_similarity_less_than) {
          filteredNeighbours.push({
            neighbor1,
            neighbor2,
            totalLength,
            cosSimilarity
          });
        }
      }
    }
    
    if (filteredNeighbours.length > 0) {
      // Find the shortest total length pair
      const shortestPair = filteredNeighbours.reduce((min, curr) => 
        curr.totalLength < min.totalLength ? curr : min
      );
      
      const { neighbor1: connection1, neighbor2: connection2, totalLength, cosSimilarity } = shortestPair;
      
      // Calculate angles with x-axis
      const thetaA = Math.atan2(connection1.normalizedPoint[1], connection1.normalizedPoint[0]) * 180 / Math.PI;
      const thetaB = Math.atan2(connection2.normalizedPoint[1], connection2.normalizedPoint[0]) * 180 / Math.PI;
      
      // Add edges to the graph
      edges.push([currentPointIndex, connection1.globalIdx]);
      edges.push([currentPointIndex, connection2.globalIdx]);
      
      // Calculate feature values for clustering
      const yDiff1 = Math.abs(connection1.normalizedPoint[1]);
      const yDiff2 = Math.abs(connection2.normalizedPoint[1]);
      const avgYDiff = (yDiff1 + yDiff2) / 2;
      
      const xDiff1 = Math.abs(connection1.normalizedPoint[0]);
      const xDiff2 = Math.abs(connection2.normalizedPoint[0]);
      const avgXDiff = (xDiff1 + xDiff2) / 2;
      
      // Calculate aspect ratio (height/width)
      const aspectRatio = avgYDiff / Math.max(avgXDiff, 0.001);
      
      // Calculate vertical alignment consistency
      const vertConsistency = Math.abs(yDiff1 - yDiff2);
      
      // Store edge properties for clustering
      edgeProperties.push([
        totalLength,
        Math.abs(thetaA + thetaB),
        aspectRatio,
        vertConsistency,
        avgYDiff
      ]);
    }
  }
  
  // Cluster the edges based on their properties
  const edgeLabels = clusterWithSingleMajority(edgeProperties);
  
  // Create a mask for edges that are not outliers (label != -1)
  const nonOutlierMask = edgeLabels.map(label => label !== -1);
  
  // Prepare the final graph structure
  const graphData = {
    nodes: points.map((point, i) => ({
      id: i,
      x: parseFloat(point[0]),
      y: parseFloat(point[1])
    })),
    edges: []
  };
  
  // Add edges with their labels, filtering out outliers
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    // Determine the corresponding edge label using division by 2 (each edge appears twice)
    const labelIndex = Math.floor(i / 2);
    const edgeLabel = edgeLabels[labelIndex];
    
    // Only add the edge if it is not an outlier
    if (nonOutlierMask[labelIndex]) {
      graphData.edges.push({
        source: parseInt(edge[0]),
        target: parseInt(edge[1]),
        label: parseInt(edgeLabel)
      });
    }
  }
  
  return graphData;
}