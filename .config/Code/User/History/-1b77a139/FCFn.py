import os
from pathlib import Path
import json

def get_obj_id(name: str, dataset_path: Path) -> int:

    with open(dataset_path/'models_eval'/'models_info.json') as models_info:
        models_dict = json.load(models_info)
        
        print(models_dict)

        names = [ dict_['name'] for key, dict_ in models_dict ]
        if name not in names:
            raise ValueError(f"Object name id not found in {dataset_path/'models_info.json'}")
