import os
import random
import torch

from torch.utils.data import Dataset
from config_paths import require_trainval_path


class CustomDataset(Dataset):
    def __init__(self, config, mode, data_list=None):
        self.config = config
        self.processed_data_path = require_trainval_path(config)

        if data_list is None:
            self.data_list_path = os.path.join(config['DATALIST']['PATH'], config['DATALIST']['NAME'][mode.upper()])
            with open(self.data_list_path, 'r') as file:
                raw_data_list = file.readlines()
        else:
            self.data_list_path = None
            raw_data_list = data_list

        self.data_list = self._filter_available_samples(raw_data_list, mode)
        if not self.data_list:
            raise FileNotFoundError(
                f"No .pt samples found for mode={mode} in {self.processed_data_path}. "
                "Generate the cache from optimized_dataset first."
            )

    def _filter_available_samples(self, data_list, mode):
        available = []
        missing_count = 0
        for file_name in data_list:
            file_name = file_name.strip()
            if not file_name:
                continue
            processed_file_name = file_name.replace('.pkl', '.pt')
            file_path = os.path.join(self.processed_data_path, processed_file_name)
            if os.path.exists(file_path):
                available.append(file_name)
            else:
                missing_count += 1

        if missing_count:
            print(
                f"CustomDataset({mode}): skipped {missing_count} entries without .pt cache "
                f"in {self.processed_data_path}."
            )
        return available

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        try:
            return self._get_item_helper(index)
        except Exception:
            fallback = random.randint(0, len(self.data_list) - 1)
            return self._get_item_helper(fallback)

    def _get_item_helper(self, index):
        file_name = self.data_list[index].strip()
        processed_file_name = file_name.replace('.pkl', '.pt')
        file_path = os.path.join(self.processed_data_path, processed_file_name)
        try:
            data = torch.load(file_path, weights_only=False)
        except TypeError:
            data = torch.load(file_path)
        return data
