import unittest
from src.data.daily_dataset import EventFeatureDataset

class TestMyDataset(unittest.TestCase):
    def test_len(self):
        dataset = EventFeatureDataset(root='.', filename='data_test.h5')
        self.assertEqual(len(dataset), 100) # assuming 100 examples in the dataset

    def test_get(self):
        dataset = MyDataset(root='.')
        data = dataset[0]
        self.assertEqual(data.x.shape, (10, 16)) # assuming features of shape (num_nodes, num_features)
        self.assertEqual(data.edge_index.shape, (2, 32)) # assuming 32 edges in the graph

if __name__ == '__main__':
    unittest.main()
