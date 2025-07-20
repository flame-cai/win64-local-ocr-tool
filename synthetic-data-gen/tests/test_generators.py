import unittest
import yaml
import random
import numpy as np
from pathlib import Path

# Add src to path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..config.config_models import Config
from src.generators.page_generator import PageGenerator
from src.generators.content_generator import ContentGenerator
from src.core.data_structures import BoxType

class TestGenerators(unittest.TestCase):

    def setUp(self):
        """Load the default config for tests."""
        config_path = Path(__file__).parent.parent / "config/default_config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = Config(**config_dict)

    def test_page_generator_reproducibility(self):
        """Test that the same seed produces the same output."""
        rand1 = random.Random(123)
        gen1 = PageGenerator(self.config, rand1)
        page1 = gen1.generate_page()[0]
        
        rand2 = random.Random(123)
        gen2 = PageGenerator(self.config, rand2)
        page2 = gen2.generate_page()[0]

        self.assertEqual(page1.points.shape, page2.points.shape)
        self.assertEqual(page1.layout_strategy, page2.layout_strategy)
        np.testing.assert_array_almost_equal(page1.points, page2.points)
        np.testing.assert_array_equal(page1.textline_labels, page2.textline_labels)

    def test_ambiguous_layout_forks(self):
        """Test that ambiguous layouts produce two page objects."""
        # Force the choice to be 'grid'
        self.config.layout_strategy_selection.choices = ["grid"]
        self.config.layout_strategy_selection.weights = [1.0]

        rand = random.Random(42)
        page_gen = PageGenerator(self.config, rand)
        pages = page_gen.generate_page()

        self.assertEqual(len(pages), 2)
        # Check that the geometry is identical
        np.testing.assert_array_equal(pages[0].points, pages[1].points)
        # Check that the textline labels are different
        self.assertFalse(np.array_equal(pages[0].textline_labels, pages[1].textline_labels))
        # Check that sample sub_ids are different
        self.assertEqual(pages[0].sub_id, 0)
        self.assertEqual(pages[1].sub_id, 1)

    def test_content_generator_alignment(self):
        """Test the alignment logic in ContentGenerator."""
        box_width = 200
        box_height = 500
        
        # Configure for predictable single line, single word output
        box_config = self.config.textbox_types['main_text']
        box_config.lines_per_box.min = 1
        box_config.lines_per_box.max = 1
        box_config.words_per_line.min = 1
        box_config.words_per_line.max = 1
        
        aug_config = self.config.augmentation_profiles['default']
        rand = random.Random(42)

        # Test LEFT alignment
        box_config.alignment.choices = ["left"]
        content_gen = ContentGenerator(box_config, aug_config, rand)
        textbox = content_gen.generate_textbox_content(box_width, box_height, BoxType.MAIN_TEXT)
        points, _ = textbox.consolidate()
        self.assertAlmostEqual(np.min(points[:, 0]), -box_width / 2, delta=5)

        # Test RIGHT alignment
        box_config.alignment.choices = ["right"]
        content_gen = ContentGenerator(box_config, aug_config, rand)
        textbox = content_gen.generate_textbox_content(box_width, box_height, BoxType.MAIN_TEXT)
        points, _ = textbox.consolidate()
        self.assertAlmostEqual(np.max(points[:, 0]), box_width / 2, delta=5)

    def test_grid_layout_generator(self):
        """Test the grid layout function directly."""
        from src.generators.layout_strategies import grid_layout
        rand = random.Random(1)
        # Force specific grid size
        self.config.layout_strategies.grid.rows.min = 5
        self.config.layout_strategies.grid.rows.max = 5
        self.config.layout_strategies.grid.cols.min = 10
        self.config.layout_strategies.grid.cols.max = 10

        points, labels = grid_layout(self.config, 1000, 1000, rand)
        
        self.assertEqual(points.shape, (50, 3))
        self.assertIn("horizontal", labels)
        self.assertIn("vertical", labels)
        self.assertEqual(labels["horizontal"].shape, (50,))
        self.assertEqual(labels["vertical"].shape, (50,))
        # Horizontal labels should be [0,0,0... 1,1,1... 4,4,4...]
        self.assertEqual(np.max(labels["horizontal"]), 4)
        # Vertical labels should be [0,1,2..9, 0,1,2..9, ...]
        self.assertEqual(np.max(labels["vertical"]), 9)

if __name__ == '__main__':
    unittest.main()