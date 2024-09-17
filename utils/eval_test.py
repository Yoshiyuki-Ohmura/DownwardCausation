import unittest
import torch
import torch.nn.functional as F
from torch.testing import assert_allclose
import eval


class ColorCodeImagesTest(unittest.TestCase):
    def test_plain_color(self):
        # setup
        inputs = torch.ones((7, 3, 3, 3), dtype=torch.float32)
        inputs = inputs * torch.tensor([[1., 0., 0.],
                                        [0., 1., 0.],
                                        [1., 1., 0.],
                                        [0., 0., 1.],
                                        [1., 0., 1.],
                                        [0., 1., 1.],
                                        [1., 1., 1.]]).view(7, 3, 1, 1)

        self.assertTrue(torch.all(
            eval.color_code_images(inputs, .5) == torch.arange(7)
        ))

    def test_images_with_bg(self):
        # setup
        inputs = torch.ones((7, 3, 3, 3), dtype=torch.float)
        inputs = inputs * torch.tensor([[1., 0., 0.],
                                        [0., 1., 0.],
                                        [1., 1., 0.],
                                        [0., 0., 1.],
                                        [1., 0., 1.],
                                        [0., 1., 1.],
                                        [1., 1., 1.]]).view(7, 3, 1, 1)
        inputs = F.pad(inputs, (1, 1, 1, 1), "constant", 0.1)

        self.assertTrue(torch.all(
            eval.color_code_images(inputs, .5) == torch.arange(7)
        ))


class ColorInvarianceTest(unittest.TestCase):
    def test_same_images(self):
        # setup
        x = torch.randn(5, 3, 10, 10)
        y = x.clone()

        self.assertAlmostEqual(eval.color_invariance(x, y), 1.)


class ShapeInvarianceTest(unittest.TestCase):
    def test_same_images(self):
        # setup
        x = torch.randn(5, 3, 10, 10)
        y = x.clone()

        assert_allclose(eval.shape_invariance(x, y),
                        torch.ones(len(x)))


if __name__ == "__main__":
    unittest.main()
