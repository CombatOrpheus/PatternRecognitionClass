from pathlib import Path
from unittest import TestCase

from optuna.terminator.improvement.emmr import torch

from src.ReachabilityGraphDataset import ReachabilityGraphDataset

DATASET_PATH = Path("../Data/GridData_DS2_test_data.processed")

class TestReachabilityGraphDataset(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance = ReachabilityGraphDataset(DATASET_PATH, 16)

    def test_create_dataloader(self):
        loader = self.instance.create_dataloader()
        self.assertIsInstance(loader, torch.utils.data.DataLoader)
        self.assertEqual(loader.batch_size, 16)

    def test_dataset_attributes(self):
        self.assertEqual(self.instance.get_num_features(), 1)
        self.assertTrue(self.instance.get_dataset_size() > 0)
        self.assertIsInstance(self.instance.get_actual_as_tensor(), torch.Tensor)