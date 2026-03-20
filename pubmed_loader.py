import json
import os

class PubMedQAData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_set = self._load_json(os.path.join(data_dir, 'pqal_fold0/train_set.json'))
        self.dev_set = self._load_json(os.path.join(data_dir, 'pqal_fold0/dev_set.json'))
        self.test_set = self._load_json(os.path.join(data_dir, 'test_set.json'))
        
        self.test_ground_truth = self._load_json(os.path.join(data_dir, 'test_ground_truth.json'))

    def _load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: File {file_path} not found.")
            return {}

    def get_sample_by_id(self, dataset_type, doc_id):
        datasets = {
            'train': self.train_set,
            'dev': self.dev_set,
            'test': self.test_set
        }
        return datasets.get(dataset_type, {}).get(doc_id)