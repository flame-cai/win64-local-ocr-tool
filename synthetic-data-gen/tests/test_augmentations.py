import unittest
import numpy as np
import random
import yaml
from pathlib import Path

# Add src to path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.augmentations import phase2_geometric, phase3_global
from src.config.config_models import Config

class TestAugmentations(unittest.TestCase):

    def setUp(self):
        """Create a simple point cloud and config for testing."""
        self.points = np.array([
            [-10, -10, 10], [10, -10, 10], [-10, 10, 10], [10, 10, 10]
        ], dtype=np.float32)
        
        config_path = Path(__file__).parent.parent / "config/default_config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = Config(**config_dict)
        self.random_state = random.Random(42)

    def test_shear(self):
        """Test the shear augmentation."""
        params = self.config.augmentation_profiles['default'].shear
        # Force a predictable shear for testing
        self.random_state.uniform = lambda a, b: 0.5 if a < b else -0.5
        
        sheared_points = phase2_geometric.shear(self.points.copy(), params, self.random_state)
        
        # Expected result for x-shear of 0.5, y-shear of 0.5
        # x' = x + 0.5*y, y' = y + 0.5*x
        expected_points = self.points.copy()
        expected_points[:, 0] = self.points[:, 0] + 0.5 * self.points[:, 1]
        expected_points[:, 1] = self.points[:, 1] + 0.5 * self.points[:, 0]
        
        np.testing.assert_array_almost_equal(sheared_points, expected_points)

    def test_stretch(self):
        """Test the stretch augmentation."""
        params = self.config.augmentation_profiles['default'].stretch
        self.random_state.uniform = lambda a, b: 1.5
        
        stretched_points = phase2_geometric.stretch(self.points.copy(), params, self.random_state)
        
        expected_points = self.points.copy()
        expected_points[:, 0] *= 1.5
        expected_points[:, 1] *= 1.5

        np.testing.assert_array_almost_equal(stretched_points, expected_points)

    def test_warp(self):
        """Test the warp augmentation to ensure it modifies points."""
        params = self.config.augmentation_profiles['default'].warp
        self.random_state.uniform = lambda a, b: (a + b) / 2 # predictable sampling
        self.random_state.choice = lambda c: 'x'

        warped_points = phase2_geometric.warp(self.points.copy(), params, self.random_state)

        # We don't test for exact values, just that something changed and shape is preserved
        self.assertEqual(self.points.shape, warped_points.shape)
        self.assertFalse(np.array_equal(self.points, warped_points))
        # Check that font size was not modified
        np.testing.assert_array_equal(self.points[:, 2], warped_points[:, 2])

    def test_point_dropout(self):
        """Test the point dropout augmentation."""
        params = self.config.augmentation_profiles['default'].point_dropout
        params.prob = 0.5 # Force a 50% dropout rate
        
        # Provide a predictable random sequence for choices
        self.random_state.choices = lambda pop, w, k: [True, False, True, False]
        
        dropped_points = phase3_global.point_dropout(self.points.copy(), params, self.random_state)
        
        self.assertEqual(dropped_points.shape[0], 2)
        np.testing.assert_array_equal(dropped_points[:, 0], np.array([-10, -10]))

    def test_global_jitter(self):
        """Test the global jitter augmentation."""
        params = self.config.augmentation_profiles['default'].global_jitter
        # Mock random.normal to be predictable
        np.random.seed(42) # global_jitter uses np.random
        
        jittered_points = phase3_global.global_jitter(self.points.copy(), params, self.random_state)

        self.assertEqual(self.points.shape, jittered_points.shape)
        self.assertFalse(np.allclose(self.points, jittered_points))
        np.testing.assert_array_equal(self.points[:, 2], jittered_points[:, 2])

if __name__ == '__main__':
    unittest.main()