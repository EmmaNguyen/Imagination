import unittest

from pytorch_deep_learning.utils.data_transform import ShepardMetzler, Scene

class TestDataTransform(unittest.TestCase):

    def test_Shepard_Metzler(self):
    #
        data_path = "/data/milatmp1/nguthien/emma/EmmaLab/DATA/3D_Scenes/Demo/shepard_metzler_7_parts-torch/train/"
        shepard_metzler = ShepardMetzler(data_path)
    #
        self.assertEqual(shepard_metzler.__len__(), 1711)
    #
    #     # file_name="1710" # Only a file name, not include extension (".pt")
    #     # for file_name in range(1711):
    #     #     try:
    #     #         image, viewpoint = shepard_metzler.__getitem__(str(file_name))
    #     #         print("\n At least one serializd file works! \n")
    #     #         break
    #     #     except AttributeError:
    #     #         pass
    #
        # file_name = "0"
        # image, viewpoint = shepard_metzler.__getitem__(file_name)

    # def test_serialization_filelike(self):
    #     # Test serialization (load and save) with a filelike object
    #     b = self._test_serialization_data()
    #     with BytesIOContext() as f:
    #         torch.save(b, f)
    #         f.seek(0)
    #         c = torch.load(f)
    #     self._test_serialization_assert(b, c)
        pass

if __name__=="__main__":
    unittest.main()
