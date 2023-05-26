import unittest
from src.data.daily_dataset import MyDataset

class TestMyDataset(unittest.TestCase):
    def test_len(self):
        dataset = MyDataset(root='.', filename='data_test.h5')
        self.assertEqual(len(dataset), 100) # assuming 100 examples in the dataset

    def test_get(self):
        dataset = MyDataset(root='.', filename='data_test.h5')
        data = dataset[0]
        self.assertEqual(data.x.shape, (1, 5)) # updated the shape according to the changes in MyDataset
        # Removed the line related to data.edge_index.shape since we don't have edge information

if __name__ == '__main__':
    unittest.main()
