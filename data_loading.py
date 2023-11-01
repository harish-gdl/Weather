import torch
from torch_geometric.data import InMemoryDataset

class NepalWeatherGraphDATASET(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NepalWeatherGraphDATASET, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Define the name of the raw file (optional)
        return []

    @property
    def processed_file_names(self):
        # Define the name of the processed file
        return ['data.pt']

    def download(self):
        # Download the dataset (optional)
        pass

    def process(self):
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_weights, y=labels)

        # Split the data into train, validation, and test sets
        num_samples = len(data)
        num_train = int(0.8 * num_samples)  # 80% for training
        num_val = int(0.1 * num_samples)  # 10% for validation
        num_test = num_samples - num_train - num_val  # Remaining for testing

        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:(num_train + num_val)]
        test_indices = indices[(num_train + num_val):]

        data.train_mask = torch.zeros(num_samples, dtype=torch.bool)
        data.val_mask = torch.zeros(num_samples, dtype=torch.bool)
        data.test_mask = torch.zeros(num_samples, dtype=torch.bool)

        data.train_mask[train_indices] = 1
        data.val_mask[val_indices] = 1
        data.test_mask[test_indices] = 1

        data_list = [data]  # In this case, we have a single data instance
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_dataset(pred_type:str):
    if pred_type=='rainfall':
        dataset = NepalWeatherGraphDATASET(root='D:/PhD_Deep_Learning/Data Rijan Sir/graphData_annotedFeat_RainTomorrow')
    elif pred_type == 'temperature':
        dataset = NepalWeatherGraphDATASET(root='D:/PhD_Deep_Learning/Data Rijan Sir/Four Prediction Data/graphDataTemperaturePrediction')
    elif pred_type == 'pressure':
        dataset = NepalWeatherGraphDATASET(root='D:/PhD_Deep_Learning/Data Rijan Sir/graphData_annotedFeat_PressureTomorrow')
    elif pred_type == 'humidity':
        dataset = NepalWeatherGraphDATASET(root='D:/PhD_Deep_Learning/Data Rijan Sir/graphData_annotedFeat_HumidityTomorrow')
    else:
        raise ValueError(f"Graph data for {pred_type}  prediction is not available")
    return dataset