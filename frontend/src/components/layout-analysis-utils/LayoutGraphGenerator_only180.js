// layoutGraphGenerator.js

/**
 * Build a KD-Tree for fast neighbor lookup
 */
class KDTree {
  constructor(points) {
    this.points = points;
    // The tree is built with objects containing the point and its original index
    this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  buildTree(points, depth) {
    if (points.length === 0) return null;
    if (points.length === 1) return points[0];

    const k = 2; // 2D points
    const axis = depth % k;
    
    // Sort points by the current axis and find the median
    points.sort((a, b) => a.point[axis] - b.point[axis]);
    const median = Math.floor(points.length / 2);
    
    // Create a node and construct subtrees
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
      
      // Add to best list if it's smaller than the farthest in the list or if list is not full
      if (best.length < k) {
        best.push({ distance, index: node.index });
        // Keep sorted descending by distance to easily find/replace the farthest point
        best.sort((a, b) => b.distance - a.distance); 
      } else if (distance < best[0].distance) {
        best[0] = { distance, index: node.index };
        best.sort((a, b) => b.distance - a.distance);
      }
      
      const axis = depth % 2;
      const diff = queryPoint[axis] - node.point[axis];
      
      const closer = diff < 0 ? node.left : node.right;
      const farther = diff < 0 ? node.right : node.left;
      
      search(closer, depth + 1);
      
      // Check if the other subtree could have closer points
      if (best.length < k || Math.abs(diff) < best[0].distance) {
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
 * Generate a graph representation of text layout based on points.
 * This function implements the simplified layout analysis logic.
 */
export function generateLayoutGraph(points) { 
  const NUM_NEIGHBOURS = 8;
  // A cosine similarity of -1.0 represents a perfect 180-degree angle.
  // This threshold selects pairs that are nearly opposite.
  const cos_similarity_less_than = -0.8;
  
  // Build a KD-tree for fast neighbor lookup
  const tree = new KDTree(points);
  
  // Store graph edges
  const edges = [];
  
  // Process nearest neighbors for each point
  for (let currentPointIndex = 0; currentPointIndex < points.length; currentPointIndex++) {
    const currentPoint = points[currentPointIndex];
    const nbrIndices = tree.query(currentPoint, NUM_NEIGHBOURS);
    
    // Skip the point itself if it's found in its neighbors
    const selfIndex = nbrIndices.indexOf(currentPointIndex);
    if (selfIndex > -1) {
        nbrIndices.splice(selfIndex, 1);
    }

    const normalizedPoints = nbrIndices.map(idx => [
      points[idx][0] - currentPoint[0],
      points[idx][1] - currentPoint[1]
    ]);
    
    const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
    const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
    
    const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
      globalIdx,
      scaledPoint: scaledPoints[i]
    }));
    
    // Find all pairs of neighbors with angles close to 180 degrees
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
        
        // If the angle is close to 180 degrees, add edges to the graph
        if (cosSimilarity < cos_similarity_less_than) {
          edges.push([currentPointIndex, neighbor1.globalIdx]);
          edges.push([currentPointIndex, neighbor2.globalIdx]);
        }
      }
    }
  }
  
  // Prepare the final graph structure
  const graphData = {
    nodes: points.map((point, i) => ({
      id: i,
      x: parseFloat(point[0]),
      y: parseFloat(point[1]),
      s: parseFloat(point[2]), // font-size
    })),
    // Add all generated edges, ensuring the 'label' property is present for compatibility.
    edges: edges.map(edge => ({
      source: parseInt(edge[0]),
      target: parseInt(edge[1]),
      // Hardcode label to 0 to match the original output format for non-outlier edges.
      label: 0 
    }))
  };
  
  return graphData;
}