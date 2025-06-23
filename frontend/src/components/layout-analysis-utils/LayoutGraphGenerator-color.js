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

// DBSCAN functions are kept in the file as they might be used by other parts,
// but they are no longer used by generateLayoutGraph.
// If they are exclusively for the old generateLayoutGraph, they can be removed.

/**
 * DBSCAN clustering implementation to identify majority cluster and outliers
 */
function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
  if (toCluster.length === 0) return [];
  
  const labels = dbscan(toCluster, eps, minSamples);
  const labelCounts = {};
  labels.forEach(label => {
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  });
  
  let majorityLabel = null;
  let maxCount = 0;
  
  for (const [label, count] of Object.entries(labelCounts)) {
    const labelNum = parseInt(label);
    if (labelNum !== -1 && count > maxCount) {
      majorityLabel = labelNum;
      maxCount = count;
    }
  }
  
  const newLabels = new Array(labels.length).fill(-1);
  if (majorityLabel !== null) {
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === majorityLabel) {
        newLabels[i] = 0;
      }
    }
  }
  return newLabels;
}

function dbscan(points, eps, minSamples) {
  const labels = new Array(points.length).fill(-1);
  let clusterId = 0;
  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue;
    const neighbors = getNeighbors(points, i, eps);
    if (neighbors.length < minSamples) {
      labels[i] = -1;
    } else {
      expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
      clusterId++;
    }
  }
  return labels;
}

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

function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
  labels[pointIndex] = clusterId;
  let i = 0;
  while (i < neighbors.length) {
    const neighborIndex = neighbors[i];
    if (labels[neighborIndex] === -1) {
      labels[neighborIndex] = clusterId;
      const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
      if (neighborNeighbors.length >= minSamples) {
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
 * MODIFIED: Removes anomaly detection, adds node edge counts and edge overlaps.
 */
export function generateLayoutGraph(points) {
  const NUM_NEIGHBOURS = 6;
  const cos_similarity_less_than = -0.8;
  
  if (!points || points.length === 0) {
    return { nodes: [], edges: [] };
  }

  const tree = new KDTree(points);
  const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
  
  const allPotentialRawEdges = [];
  
  for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
    const nbrIndices = indices[currentPointIndex].filter(idx => idx !== currentPointIndex); // Exclude self
    if (nbrIndices.length < 2) continue; // Need at least two neighbors to form a line segment

    const currentPoint = points[currentPointIndex];
    
    const normalizedPoints = nbrIndices.map(idx => [
      points[idx][0] - currentPoint[0],
      points[idx][1] - currentPoint[1]
    ]);
    
    // Scaling factor computation can be simplified or removed if not strictly necessary
    // For this version, let's keep it consistent with original logic path for now.
    const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
    const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
    
    const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
      globalIdx,
      scaledPoint: scaledPoints[i],
      normalizedPoint: normalizedPoints[i]
    }));
    
    const filteredPairCandidates = [];
    
    for (let i = 0; i < relativeNeighbours.length; i++) {
      for (let j = i + 1; j < relativeNeighbours.length; j++) {
        const neighbor1 = relativeNeighbours[i];
        const neighbor2 = relativeNeighbours[j];
        
        const norm1Scaled = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
        const norm2Scaled = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
        
        let cosSimilarity = 0.0;
        if (norm1Scaled * norm2Scaled !== 0) {
          const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                           neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
          cosSimilarity = dotProduct / (norm1Scaled * norm2Scaled);
        }
        
        if (cosSimilarity < cos_similarity_less_than) {
          const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
          const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
          const totalLength = norm1Real + norm2Real;

          filteredPairCandidates.push({
            neighbor1,
            neighbor2,
            totalLength,
            // cosSimilarity // Not strictly needed beyond filtering
          });
        }
      }
    }
    
    if (filteredPairCandidates.length > 0) {
      const shortestPair = filteredPairCandidates.reduce((min, curr) => 
        curr.totalLength < min.totalLength ? curr : min
      );
      
      const { neighbor1: connection1, neighbor2: connection2 } = shortestPair;
      
      // Add directed edges based on this finding. Overlaps will be counted later.
      allPotentialRawEdges.push({ u: currentPointIndex, v: connection1.globalIdx });
      allPotentialRawEdges.push({ u: currentPointIndex, v: connection2.globalIdx });
    }
  }
  
  // Process raw edges to create unique undirected edges with overlap counts
  const finalEdgesMap = new Map(); // Key: "minIdx_maxIdx", Value: count (overlap)
  
  for (const rawEdge of allPotentialRawEdges) {
    const u = rawEdge.u;
    const v = rawEdge.v;
    if (u === v) continue; // Skip self-loops if any were accidentally generated

    const s = Math.min(u, v);
    const t = Math.max(u, v);
    const key = `${s}_${t}`;
    
    finalEdgesMap.set(key, (finalEdgesMap.get(key) || 0) + 1);
  }
  
  const finalEdges = [];
  for (const [key, count] of finalEdgesMap.entries()) {
    const [s, t] = key.split('_').map(Number);
    finalEdges.push({
      source: s,
      target: t,
      overlaps: count
      // No 'label' property from anomaly detection anymore
    });
  }

  // Calculate number of edges for each node (degree)
  const nodeDegrees = new Array(points.length).fill(0);
  for (const edge of finalEdges) {
    nodeDegrees[edge.source]++;
    nodeDegrees[edge.target]++;
  }
  
  const graphData = {
    nodes: points.map((point, i) => ({
      id: i,
      x: parseFloat(point[0]),
      y: parseFloat(point[1]),
      s: parseFloat(point[2]), // Font size
      numEdges: nodeDegrees[i]  // New property: number of edges
    })),
    edges: finalEdges // Edges now have 'overlaps' property
  };
  
  return graphData;
}