"""Small analytic checks for attribution semantics, coverage, and geometry."""

import unittest
import numpy as np
from PIL import Image
from lwl_models.ablation import patch_boxes, ablated_images, aggregate_patch_effects, score_effects
from lwl_models.low_level import low_level_saliency


class AblationTests(unittest.TestCase):
    def test_edge_coverage_and_overlap_normalization(self):
        boxes = patch_boxes((8, 7), patch_size=3, stride=2)
        array, coverage = aggregate_patch_effects(boxes, np.full(len(boxes), 2.0), (8, 7))
        np.testing.assert_array_equal(array, np.full((7, 8), 2.0))
        self.assertTrue(np.all(coverage > 0))
        self.assertGreater(coverage.max(), coverage.min())
        self.assertEqual(boxes[:, 2].max(), 8)
        self.assertEqual(boxes[:, 3].max(), 7)

    def test_reject_uncovered_grid(self):
        with self.assertRaises(ValueError):
            patch_boxes((10, 10), patch_size=2, stride=3)

    def test_competitor_margin_and_signed_evidence(self):
        intact = np.array([5.0, 3.0, 1.0])
        ablated = np.array([[3.0, 2.0, 1.0], [6.0, 0.0, 2.0]])
        result = score_effects(intact, ablated, 0, {"Related": 1, "Unrelated": 2})
        np.testing.assert_array_equal(result["recognition"], [2.0, -1.0])
        np.testing.assert_array_equal(result["contrast_related"], [1.0, -4.0])
        np.testing.assert_array_equal(result["contrast_unrelated"], [2.0, 0.0])
        # A word contrasted against itself must give no discriminative evidence.
        same = score_effects(intact, ablated, 0, {"Same": 0})
        np.testing.assert_array_equal(same["contrast_same"], [0.0, 0.0])

    def test_blur_changes_only_selected_region(self):
        checker = ((np.indices((12, 12)).sum(axis=0) % 2) * 255).astype(np.uint8)
        original = np.repeat(checker[..., None], 3, axis=2)
        image = Image.fromarray(original)
        changed = np.asarray(next(ablated_images(image, [(3, 4, 8, 9)], blur_radius=2)))
        outside = np.ones((12, 12), dtype=bool)
        outside[4:9, 3:8] = False
        np.testing.assert_array_equal(changed[outside], original[outside])
        self.assertTrue(np.any(changed[~outside] != original[~outside]))
        np.testing.assert_array_equal(np.asarray(image), original)


class LowLevelTests(unittest.TestCase):
    def test_uniform_images_have_no_conspicuity(self):
        for color in [(0, 0, 0), (128, 128, 128), (255, 0, 0)]:
            result = low_level_saliency(Image.new("RGB", (128, 128), color))
            for array in result.values():
                np.testing.assert_allclose(array, 0, atol=1e-10)

    def test_local_contrast_produces_finite_image_sized_map(self):
        rgb = np.full((128, 128, 3), 80, dtype=np.uint8)
        rgb[30:50, 60:80] = [255, 255, 255]
        result = low_level_saliency(Image.fromarray(rgb))
        array = result["low_level"]
        self.assertEqual(array.shape, (128, 128))
        self.assertTrue(np.isfinite(array).all())
        self.assertGreater(array.max(), 0)
        self.assertGreaterEqual(array.min(), 0)
        self.assertLessEqual(array.max(), 1)


class GeometryTests(unittest.TestCase):
    def test_map_and_image_keep_the_same_upright_extent(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from lwl_models.plotting import draw_map
        from lwl_models.study import STIM_BOX
        image = Image.new("RGB", (500, 500), "white")
        array = np.zeros((500, 500))
        array[:100] = 1  # Evidence near the TOP of the image must stay at the top.
        fig, ax = plt.subplots()
        draw_map(ax, image, array, "magma", Normalize(0, 1))
        self.assertFalse(ax.yaxis_inverted())
        for artist in ax.images:
            self.assertEqual(tuple(artist.get_extent()), STIM_BOX)
            self.assertEqual(artist.origin, "upper")
        self.assertEqual(ax.patches[0].get_width(), 620)
        self.assertEqual(STIM_BOX[1] - STIM_BOX[0], 500)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
