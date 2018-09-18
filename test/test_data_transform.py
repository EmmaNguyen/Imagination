import unittest

from pytorch_deep_learning.utils import data_transform

class TestDataTransform(unittest.TestCase):

    def test_Shepard_Metzler(self):
        try:
            data_path = "./data/small_shepard_metzler_7_parts/train"
            shepard_metzler = data_loader.ShepardMetzler(data_path)
            self.assertEqual(shepard_metzler.__len__(), 1711)
        except:
            print("Check data path. Make sure that we have the same location!")

if __name__=="__main__":
    unittest.main()
