import unittest
import numpy as np

# Add src to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_structures import Point, Word, TextLine, TextBox, TextBoxBlueprint, BoxType

class TestDataStructures(unittest.TestCase):

    def test_consolidation_simple(self):
        """Test the TextBox.consolidate() method with a simple structure."""
        # Create a nested structure of objects
        p1 = Point(x=1, y=1, font_size=10)
        p2 = Point(x=2, y=1, font_size=10)
        w1 = Word(points=[p1, p2])
        tl1 = TextLine(words=[w1])
        
        p3 = Point(x=1, y=5, font_size=12)
        w2 = Word(points=[p3])
        tl2 = TextLine(words=[w2])
        
        textbox = TextBox(text_lines=[tl1, tl2], box_type=BoxType.MAIN_TEXT)
        
        # Consolidate into NumPy arrays
        points_local, line_ids_local = textbox.consolidate()
        
        # Check shapes
        self.assertEqual(points_local.shape, (3, 3))
        self.assertEqual(line_ids_local.shape, (3,))
        
        # Check content
        expected_points = np.array([
            [1, 1, 10], [2, 1, 10], # Line 0
            [1, 5, 12]             # Line 1
        ], dtype=np.float32)
        
        expected_line_ids = np.array([0, 0, 1], dtype=np.int32)
        
        np.testing.assert_array_almost_equal(points_local, expected_points)
        np.testing.assert_array_equal(line_ids_local, expected_line_ids)

    def test_consolidation_complex(self):
        """Test consolidation with multiple words per line."""
        p1, p2 = Point(1,1,8), Point(2,1,8)
        w1 = Word([p1, p2])
        p3 = Point(5,1,8)
        w2 = Word([p3])
        tl1 = TextLine([w1, w2]) # Line 0

        p4 = Point(1,10,8)
        w3 = Word([p4])
        tl2 = TextLine([w3]) # Line 1

        textbox = TextBox([tl1, tl2], BoxType.MARGINALIA)
        points, lines = textbox.consolidate()

        self.assertEqual(points.shape, (4, 3))
        expected_lines = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(lines, expected_lines)

    def test_empty_textbox_consolidation(self):
        """Test consolidation of a textbox with no content."""
        textbox = TextBox(text_lines=[], box_type=BoxType.MAIN_TEXT)
        points, lines = textbox.consolidate()
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(lines.shape, (0,))

    def test_empty_line_consolidation(self):
        """Test consolidation of a textbox with an empty textline."""
        tl1 = TextLine(words=[Word([Point(1,1,10)])])
        tl2 = TextLine(words=[]) # Empty line
        tl3 = TextLine(words=[Word([Point(2,2,10)])])
        textbox = TextBox([tl1, tl2, tl3], BoxType.MAIN_TEXT)
        points, lines = textbox.consolidate()
        
        self.assertEqual(points.shape, (2, 3))
        expected_lines = np.array([0, 2])
        np.testing.assert_array_equal(lines, expected_lines)

    def test_blueprint_obb_no_rotation(self):
        """Test OBB generation for a blueprint with zero rotation."""
        bp = TextBoxBlueprint(
            box_type=BoxType.MAIN_TEXT,
            position=(100, 200),
            width=50,
            height=30,
            orientation_deg=0
        )
        obb = bp.get_obb()
        
        expected_obb = np.array([
            [100 - 25, 200 - 15],  # Top-left
            [100 + 25, 200 - 15],  # Top-right
            [100 + 25, 200 + 15],  # Bottom-right
            [100 - 25, 200 + 15]   # Bottom-left
        ])
        np.testing.assert_array_almost_equal(obb, expected_obb)
        
    def test_blueprint_obb_with_rotation(self):
        """Test OBB generation for a blueprint with 90-degree rotation."""
        bp = TextBoxBlueprint(
            box_type=BoxType.MAIN_TEXT,
            position=(100, 200),
            width=50,
            height=30,
            orientation_deg=90
        )
        obb = bp.get_obb()
        
        # After 90 deg rotation, a corner at (-25, -15) becomes (15, -25)
        # Then translated by (100, 200) -> (115, 175)
        expected_obb = np.array([
            [100 + 15, 200 - 25],
            [100 + 15, 200 + 25],
            [100 - 15, 200 + 25],
            [100 - 15, 200 - 25]
        ])
        np.testing.assert_array_almost_equal(obb, expected_obb)

if __name__ == '__main__':
    unittest.main()