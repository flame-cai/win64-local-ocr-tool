import unittest
import numpy as np

# Add src to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.coordinate_systems import transform_to_global, normalize_page_coordinates
from src.core.data_structures import TextBoxBlueprint, BoxType

class TestCoordinateSystems(unittest.TestCase):

    def test_transform_to_global_translation_only(self):
        """Test global transformation with only translation (zero rotation)."""
        points_local = np.array([[0, 0, 10], [5, 10, 10]], dtype=np.float32)
        blueprint = TextBoxBlueprint(
            box_type=BoxType.MAIN_TEXT,
            position=(100, 200),
            width=10, height=20, orientation_deg=0
        )
        
        points_global = transform_to_global(points_local, blueprint)
        
        expected_points = np.array([[100, 200, 10], [105, 210, 10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(points_global, expected_points)

    def test_transform_to_global_with_rotation(self):
        """Test global transformation with 90-degree rotation and translation."""
        points_local = np.array([[10, 20, 10]], dtype=np.float32) # A single point
        blueprint = TextBoxBlueprint(
            box_type=BoxType.MAIN_TEXT,
            position=(100, 200),
            width=10, height=20, orientation_deg=90
        )
        
        points_global = transform_to_global(points_local, blueprint)
        
        # Rotation: (10, 20) -> (-20, 10)
        # Translation: (-20, 10) + (100, 200) -> (80, 210)
        expected_points = np.array([[80, 210, 10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(points_global, expected_points)

    def test_transform_empty_points(self):
        """Test that transformation handles empty point clouds gracefully."""
        points_local = np.empty((0, 3), dtype=np.float32)
        blueprint = TextBoxBlueprint(
            box_type=BoxType.MAIN_TEXT,
            position=(100, 200),
            width=10, height=20, orientation_deg=0
        )
        points_global = transform_to_global(points_local, blueprint)
        self.assertEqual(points_global.shape, (0, 3))

    def test_normalize_coordinates(self):
        """Test the page normalization logic."""
        # Note: generation uses a centered coordinate system, so we test with that assumption
        # Points are placed around (0,0) and then shifted by page_dim/2 for normalization
        page_width, page_height = 1000, 1500 # height is the longest dimension
        points_global = np.array([
            [0, 0, 30],             # Center point
            [-500, -750, 15],       # Top-left corner of page bounds
            [500, 750, 15],         # Bottom-right corner of page bounds
        ])

        normalized_points = normalize_page_coordinates(points_global, page_width, page_height)
        
        # Expected after shifting origin and dividing by longest_dim (1500)
        # x' = (x + 1000/2) / 1500; y' = (y + 1500/2) / 1500; fs' = fs / 1500
        expected_normalized = np.array([
            [(0 + 500)/1500, (0 + 750)/1500, 30/1500],
            [(-500 + 500)/1500, (-750 + 750)/1500, 15/1500],
            [(500 + 500)/1500, (750 + 750)/1500, 15/1500],
        ])

        np.testing.assert_array_almost_equal(normalized_points, expected_normalized)
        
        # Check that one of the normalized spatial dimensions has a max of 1.0
        self.assertAlmostEqual(np.max(normalized_points[:, 1]), 1.0)

if __name__ == '__main__':
    unittest.main()