import unittest

from pytorch_deep_learning.utils import data_transform

class TestDataTransform(unittest.TestCase):

    def test_Shepard_Metzler(self):
        data_path = "/data/milatmp1/nguthien/emma/EmmaLab/DATA/3D_Scenes/Demo/shepard_metzler_7_parts-torch/train"

        shepard_metzler = data_loader.ShepardMetzler(data_path)
        self.assertEqual(shepard_metzler.__len__(), 1711)



if __name__=="__main__":
    unittest.main()
