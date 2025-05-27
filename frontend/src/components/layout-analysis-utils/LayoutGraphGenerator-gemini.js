// layoutGraphGenerator.js

// --- Corrected KDTree ---
class KDTree {
  constructor(points, pointObjects = null) { // Allow associating original objects
    this.pointObjects = pointObjects; // Store if provided
    // points here is an array of coordinate arrays, e.g., [[x1,y1], [x2,y2], ...]
    // pointObjects (if provided) is an array of the actual objects these coordinates refer to,
    // and should have the same length as points.
    this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  buildTree(items, depth) { // items are {point: [x,y], index: original_array_index}
    if (items.length === 0) return null;
    
    // *** CORRECTED LEAF NODE HANDLING ***
    if (items.length === 1) {
      return { 
        item: items[0], // items[0] is the {point, index} object
        left: null, 
        right: null, 
        axis: depth % 2 
      };
    }

    const k = 2; // 2D points
    const axis = depth % k;
    
    items.sort((a, b) => a.point[axis] - b.point[axis]);
    const median = Math.floor(items.length / 2);
    
    return {
      item: items[median], // Store the whole item {point, index} for the current node
      left: this.buildTree(items.slice(0, median), depth + 1),
      right: this.buildTree(items.slice(median + 1), depth + 1),
      axis: axis
    };
  }

  queryRadius(queryPoint, radius) {
    const neighborsFound = []; // Changed variable name to avoid conflict if used globally
    const radiusSq = radius * radius;

    const search = (node, depth) => {
        if (!node) return; // Base case: empty subtree

        // node.item is {point, index}
        const distanceSq = this.euclideanDistanceSq(queryPoint, node.item.point);
        if (distanceSq <= radiusSq) {
            // node.item.index is the original index into the `points` array passed to constructor
            // If pointObjects was provided, use it to return the actual object
            neighborsFound.push(this.pointObjects ? this.pointObjects[node.item.index] : node.item.index);
        }

        const axis = node.axis; // Use stored axis from the node
        const diff = queryPoint[axis] - node.item.point[axis];

        const closerSubtree = diff < 0 ? node.left : node.right;
        const fartherSubtree = diff < 0 ? node.right : node.left;

        search(closerSubtree, depth + 1);

        // Check if the hypersphere (circle in 2D) crosses the splitting plane
        if (Math.abs(diff) < radius) { 
            search(fartherSubtree, depth + 1);
        }
    };

    search(this.tree, 0);
    return neighborsFound;
  }
  
  queryKNearest(queryPoint, k) {
    const best = []; // Stores {distance, index (or object)}
    
    const search = (node, depth) => {
      if (!node) return; // Base case: empty subtree
      
      // node.item is {point, index}
      const distance = this.euclideanDistance(queryPoint, node.item.point);
      const currentPointData = this.pointObjects ? this.pointObjects[node.item.index] : node.item.index;
      
      if (best.length < k) {
        best.push({ distance, data: currentPointData });
        best.sort((a, b) => a.distance - b.distance); // Keep sorted by distance
      } else if (distance < best[best.length - 1].distance) {
        best[best.length - 1] = { distance, data: currentPointData };
        best.sort((a, b) => a.distance - b.distance);
      }
      
      const axis = node.axis; // Use stored axis from the node
      const diff = queryPoint[axis] - node.item.point[axis];
      
      const closerSubtree = diff < 0 ? node.left : node.right;
      const fartherSubtree = diff < 0 ? node.right : node.left;
      
      search(closerSubtree, depth + 1);
      
      // Check if the hypersphere containing current k-best could cross the splitting plane
      if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
        search(fartherSubtree, depth + 1);
      }
    };
    
    search(this.tree, 0);
    // queryKNearest in the main code expects an array of {index, distance} or just indices/objects.
    // Let's make it return the data (object or index) directly.
    // The original code `mainKDTree.queryKNearest(points[i], K_CANDIDATE_NEIGHBORS + 1);`
    // and then `neighbor.index` implies it needs the index property from the returned object.
    // So, we should return objects that have an `index` property (and optionally `distance`).
    
    // Return objects with {distance, index} structure, consistent with original KDTree if no pointObjects
    // If pointObjects, it's more complex. The calling code expects .index.
    // For now, stick to returning objects that have an .index for mainKDTree, and actual objects for midpointKDTree.
    
    // If this.pointObjects exists, queryKNearest was used on midpointKDTree, which expects objects.
    // If !this.pointObjects, it was used on mainKDTree, which expects {index: ..., distance: ...}
    if (this.pointObjects) {
        return best.map(b => b.data); // Returns the actual objects (e.g., candidate_edge objects)
    } else {
        // For mainKDTree, the original code iterates and uses `neighbor.index`.
        // `node.item.index` is the index in the original `points` array.
        // So, the `data` field in `best` will be this index. We need to map it to an object with an `index` field.
        return best.map(b => ({ index: b.data, distance: b.distance }));
    }
  }

  euclideanDistance(p1, p2) {
    return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
  }
  euclideanDistanceSq(p1, p2) { // For radius search, square of distance is often enough
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2;
  }
}


// --- Helper Functions (from previous response, unchanged) ---
function getLineOrientation(p1_coords, p2_coords) {
  const dx = p2_coords[0] - p1_coords[0];
  const dy = p2_coords[1] - p1_coords[1];
  let angle = Math.atan2(dy, dx); // angle in [-PI, PI]
  if (angle < 0) angle += Math.PI; // Normalize to [0, PI) for orientation
  if (angle >= Math.PI) angle = 0; // Ensure it's strictly < PI for binning
  return angle;
}

function getAngularDistance(angle1, angle2) { // angles in [0, PI)
  const diff = Math.abs(angle1 - angle2);
  return Math.min(diff, Math.PI - diff);
}

function findDominantOrientation(orientations, numBins) {
    if (!orientations || orientations.length === 0) return null;

    const bins = new Array(numBins).fill(0);
    const binWidth = Math.PI / numBins;

    for (const orient of orientations) {
        let binIndex = Math.floor(orient / binWidth);
        binIndex = Math.max(0, Math.min(numBins - 1, binIndex)); // Clamp index
        bins[binIndex]++;
    }

    let maxCount = -1;
    let maxBinIndex = -1;
    for (let i = 0; i < numBins; i++) {
        if (bins[i] > maxCount) {
            maxCount = bins[i];
            maxBinIndex = i;
        }
    }
    if (maxBinIndex === -1) return null;
    return (maxBinIndex + 0.5) * binWidth; // Center of the most populated bin
}

function dotProduct(v1, v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

function normalizeVector(v) {
    const mag = Math.sqrt(v[0]**2 + v[1]**2);
    if (mag === 0) return [0,0];
    return [v[0]/mag, v[1]/mag];
}


/**
 * Generate a graph representation of text layout based on "normative influence".
 * (This function body is from the previous "Normative Influence" response and should now work with the corrected KDTree)
 */
export function generateLayoutGraph(points) {
  if (!points || points.length < 2) {
    return { nodes: points.map((p,i) => ({id: i, x: p[0], y: p[1]})), edges: [] };
  }

  // --- Parameters ---
  const K_CANDIDATE_NEIGHBORS = 10; 
  const RADIUS_INFLUENCE_PERCENTAGE_OF_AVG_NN_DIST = 500; 
  const MIN_PEERS_FOR_CONSENSUS = 3; 
  const NUM_ORIENTATION_BINS = 18;   
  const CONFORMITY_ANGLE_TOLERANCE = Math.PI / 12; 
  const OPPOSITE_DIRECTION_COSINE_THRESHOLD = -0.8; 
  const LENGTH_PENALTY_WEIGHT = 0.1; 

  // --- Phase 1: Generate Candidate Edges ---
  const mainKDTree = new KDTree(points); // Not passing pointObjects, so queryKNearest will return {index, distance}
  let all_candidate_edges = [];
  let candidate_edge_id_counter = 0;
  const candidateEdgeMap = new Map(); 

  let totalNNDist = 0;
  let nnCount = 0;
  for (let i = 0; i < points.length; i++) {
      const neighbors = mainKDTree.queryKNearest(points[i], 2); // Expects [{index, distance}, ...]
      if (neighbors.length > 1) {
          const neighborQueryResult = neighbors[1]; // This is the closest distinct neighbor
          // neighbors[0] will be the point itself if distance is 0, or closest if not self.
          // Assuming queryKNearest returns sorted and includes self if k is large enough or self is closest.
          // If self is always first with dist 0:
          if (neighborQueryResult.index !== i || (neighbors.length > 0 && neighbors[0].index !== i && neighbors.length ===1) ) { // Ensure it's not the point itself
            totalNNDist += mainKDTree.euclideanDistance(points[i], points[neighborQueryResult.index]);
            nnCount++;
          } else if (neighbors.length > 0 && neighbors[0].index !== i) { 
             totalNNDist += mainKDTree.euclideanDistance(points[i], points[neighbors[0].index]);
             nnCount++;
          }
      } else if (neighbors.length === 1 && neighbors[0].index !== i) { // Only one point, not self
            totalNNDist += mainKDTree.euclideanDistance(points[i], points[neighbors[0].index]);
            nnCount++;
      }
  }
  const avgNNDist = nnCount > 0 ? totalNNDist / nnCount : 10; 
  const RADIUS_INFLUENCE = (avgNNDist * RADIUS_INFLUENCE_PERCENTAGE_OF_AVG_NN_DIST) / 100;


  for (let i = 0; i < points.length; i++) {
    // queryKNearest on mainKDTree returns array of {index: neighbor_original_index, distance: d}
    const neighbors = mainKDTree.queryKNearest(points[i], K_CANDIDATE_NEIGHBORS + 1); 
    for (const neighbor of neighbors) { // neighbor is {index, distance}
      const j = neighbor.index;
      if (i === j) continue;

      const p1_idx = Math.min(i, j);
      const p2_idx = Math.max(i, j);
      const edgeKey = `${p1_idx}-${p2_idx}`;

      if (!candidateEdgeMap.has(edgeKey)) {
        const p1_coords = points[p1_idx];
        const p2_coords = points[p2_idx];
        const edge = {
          id: candidate_edge_id_counter++,
          p1_idx, p2_idx,
          p1_coords, p2_coords,
          length: mainKDTree.euclideanDistance(p1_coords, p2_coords), // or neighbor.distance if i is p1_idx
          orientation: getLineOrientation(p1_coords, p2_coords),
          midpoint: [(p1_coords[0] + p2_coords[0]) / 2, (p1_coords[1] + p2_coords[1]) / 2],
          conformity_score: 0 
        };
        candidateEdgeMap.set(edgeKey, edge);
        all_candidate_edges.push(edge);
      }
    }
  }
  
  if (all_candidate_edges.length === 0) {
    return { nodes: points.map((p,i) => ({id: i, x: p[0], y: p[1]})), edges: [] };
  }

  // --- Phase 2: Score Candidate Edges based on Local Orientation Coherence ---
  const midpointCoords = all_candidate_edges.map(edge => edge.midpoint);
  // For midpointKDTree, pointObjects is all_candidate_edges.
  // queryRadius will return actual edge objects from all_candidate_edges.
  const midpointKDTree = new KDTree(midpointCoords, all_candidate_edges); 

  for (let i = 0; i < all_candidate_edges.length; i++) {
    const edge_c = all_candidate_edges[i];
    // queryRadius on midpointKDTree returns an array of edge objects
    const peer_edge_objects = midpointKDTree.queryRadius(edge_c.midpoint, RADIUS_INFLUENCE);
    
    const peer_orientations = [];
    for (const peer_edge_obj of peer_edge_objects) { 
        if (peer_edge_obj.id !== edge_c.id) {
            peer_orientations.push(peer_edge_obj.orientation);
        }
    }

    if (peer_orientations.length < MIN_PEERS_FOR_CONSENSUS) {
      edge_c.conformity_score = 0.1; 
    } else {
      const dominant_orientation = findDominantOrientation(peer_orientations, NUM_ORIENTATION_BINS);
      if (dominant_orientation === null) {
         edge_c.conformity_score = 0.1; 
      } else {
        const angular_diff = getAngularDistance(edge_c.orientation, dominant_orientation);
        edge_c.conformity_score = Math.exp(-(angular_diff ** 2) / (2 * (CONFORMITY_ANGLE_TOLERANCE ** 2)));
      }
    }
  }

  // --- Phase 3: Select Final Edges to Form Text Lines ---
  const point_to_incident_candidates = Array(points.length).fill(null).map(() => []);
  for (const cand_edge of all_candidate_edges) {
    point_to_incident_candidates[cand_edge.p1_idx].push({
      target_idx: cand_edge.p2_idx,
      orientation: cand_edge.orientation, 
      length: cand_edge.length,
      conformity_score: cand_edge.conformity_score,
      original_edge_id: cand_edge.id
    });
    point_to_incident_candidates[cand_edge.p2_idx].push({
      target_idx: cand_edge.p1_idx,
      orientation: cand_edge.orientation, 
      length: cand_edge.length,
      conformity_score: cand_edge.conformity_score,
      original_edge_id: cand_edge.id
    });
  }

  const chosen_connections_per_point = Array(points.length).fill(null).map(() => []);

  for (let i = 0; i < points.length; i++) {
    const incident_candidates = point_to_incident_candidates[i];
    if (incident_candidates.length < 2) continue;

    let best_pair = null;
    let max_objective_score = -Infinity;

    for (let j = 0; j < incident_candidates.length; j++) {
      for (let k = j + 1; k < incident_candidates.length; k++) {
        const cand1 = incident_candidates[j];
        const cand2 = incident_candidates[k];

        if (cand1.target_idx === cand2.target_idx) continue; 

        const p_i = points[i];
        const p_t1 = points[cand1.target_idx];
        const p_t2 = points[cand2.target_idx];

        const v1 = [p_t1[0] - p_i[0], p_t1[1] - p_i[1]];
        const v2 = [p_t2[0] - p_i[0], p_t2[1] - p_i[1]];

        const norm_v1 = normalizeVector(v1);
        const norm_v2 = normalizeVector(v2);
        const cos_similarity = dotProduct(norm_v1, norm_v2);

        if (cos_similarity < OPPOSITE_DIRECTION_COSINE_THRESHOLD) {
          const objective_score = (cand1.conformity_score + cand2.conformity_score)
                                 - LENGTH_PENALTY_WEIGHT * (cand1.length + cand2.length);
          if (objective_score > max_objective_score) {
            max_objective_score = objective_score;
            best_pair = { conn1: cand1, conn2: cand2 };
          }
        }
      }
    }

    if (best_pair) {
      chosen_connections_per_point[i].push(best_pair.conn1);
      chosen_connections_per_point[i].push(best_pair.conn2);
    }
  }

  // --- Resolve and Finalize Edges (Mutual Agreement) ---
  const final_graph_edges = [];
  const added_edge_ids = new Set();

  for (let i = 0; i < points.length; i++) {
    for (const chosen_conn_from_i of chosen_connections_per_point[i]) {
      const target_j = chosen_conn_from_i.target_idx;
      const edge_id = chosen_conn_from_i.original_edge_id;

      if (added_edge_ids.has(edge_id)) continue;

      for (const chosen_conn_from_j of chosen_connections_per_point[target_j]) {
        if (chosen_conn_from_j.target_idx === i && chosen_conn_from_j.original_edge_id === edge_id) {
          final_graph_edges.push({
            source: i,
            target: target_j,
            label: 0 
          });
          added_edge_ids.add(edge_id);
          break; 
        }
      }
    }
  }

  // --- Phase 4: Output ---
  return {
    nodes: points.map((point, idx) => ({
      id: idx,
      x: parseFloat(point[0]),
      y: parseFloat(point[1])
    })),
    edges: final_graph_edges
  };
}