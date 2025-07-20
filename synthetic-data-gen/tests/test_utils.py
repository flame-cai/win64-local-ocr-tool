import unittest
import numpy as np
import random
from pathlib import Path

# Add src to path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.distribution_sampler import sample_from_config
from src.utils.collision_detection import check_collision_sat
from src.core.data_structures import TextBoxBlueprint, BoxType

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.random_state = random.Random(42)

    def test_sample_from_config_uniform(self):
        config_dict = {"distribution": "uniform", "min": 10, "max": 20}
        value = sample_from_config(config_dict, self.random_state)
        self.assertGreaterEqual(value, 10)
        self.assertLessEqual(value, 20)

    def test_sample_from_config_normal(self):
        config_dict = {"distribution": "normal", "mean": 100, "std": 5}
        value = sample_from_config(config_dict, self.random_state)
        # It's statistical, so we just check if it's a float
        self.assertIsInstance(value, float)

    def test_sample_from_config_choice(self):
        config_dict = {"distribution": "choice", "choices": ["a", "b", "c"]}
        value = sample_from_config(config_dict, self.random_state)
        self.assertIn(value, ["a", "b", "c"])
        
    def test_sample_from_config_invalid(self):
        config_dict = {"distribution": "gamma"}
        with self.assertRaises(ValueError):
            sample_from_config(config_dict, self.random_state)

    def test_collision_sat_no_collision(self):
        """Test two boxes that are far apart."""
        bp1 = TextBoxBlueprint(BoxType.MAIN_TEXT, (0, 0), 10, 10, 0)
        bp2 = TextBoxBlueprint(BoxType.MAIN_TEXT, (100, 100), 10, 10, 0)
        self.assertFalse(check_collision_sat(bp1.get_obb(), bp2.get_obb()))

    def test_collision_sat_clear_collision(self):
        """Test two boxes that clearly overlap."""
        bp1 = TextBoxBlueprint(BoxType.MAIN_TEXT, (0, 0), 20, 20, 0)
        bp2 = TextBoxBlueprint(BoxType.MAIN_TEXT, (5, 5), 20, 20, 0)
        self.assertTrue(check_collision_sat(bp1.get_obb(), bp2.get_obb()))

    def test_collision_sat_touching_edge(self):
        """Test two boxes touching at an edge."""
        bp1 = TextBoxBlueprint(BoxType.MAIN_TEXT, (0, 0), 20, 20, 0)
        bp2 = TextBoxBlueprint(BoxType.MAIN_TEXT, (20, 0), 20, 20, 0)
        # SAT considers touching as collision
        self.assertTrue(check_collision_sat(bp1.get_obb(), bp2.get_obb()))
        
    def test_collision_sat_rotated(self):
        """Test collision with one box rotated."""
        bp1 = TextBoxBlueprint(BoxType.MAIN_TEXT, (0, 0), 20, 20, 0)
        bp2 = TextBoxBlueprint(BoxType.MAIN_TEXT, (15, 15), 20, 20, 45)
        self.assertTrue(check_collision_sat(bp1.get_obb(), bp2.get_obb()))
        
    def test_no_collision_sat_rotated(self):
        """Test no collision with one box rotated."""
        bp1 = TextBoxBlueprint(BoxType.MAIN_TEXT, (0, 0), 10, 10, 0)
        bp2 = TextBoxBlueprint(BoxType.MAIN_TEXT, (20, 20), 10, 10, 45)
        self.assertFalse(check_collision_sat(bp1.get_obb(), bp2.get_obb()))


if __name__ == '__main__':
    unittest.main()