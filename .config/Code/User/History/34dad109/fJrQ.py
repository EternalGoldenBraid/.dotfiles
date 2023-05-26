import pytest
from src.data.daily_dataset import EventFeatureDataset

def test_len():
    print("Testing len of dataset")
    dataset = EventFeatureDataset(root='.', filename='data_test.h5')
    assert len(dataset) == 100 # assuming 100 examples in the dataset

def test_get():
    dataset = EventFeatureDataset(root='.', filename='data_test.h5')
    data = dataset[0]
    assert data.x.shape == (1, 5) # updated the shape according to the changes in MyDataset
    # Removed the line related to data.edge_index.shape since we don't have edge information
