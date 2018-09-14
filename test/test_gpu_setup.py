"""
A test to confirm that we have the same kind of computing resources.
"""

import unittest

import torch

class TestGPUSetup(unittest.TestCase):

    def test_enable_GPU(self):
        self.assertTrue(torch.cuda.is_available())

    def test_num_device(self):
        self.assertEqual(torch.cuda.device_count(), 1)

    def name_device_test(self):
        None
        # Should recommend using M40 or P100

if __name__=="__main__":
    unittest.main()
