import argparse
import json
import os
from typing import Any, Dict

import pandas as pd
import torch

from src.lightning_models.mlp import MLPClassifier
from src.data.data_processing import get_num_labels
from src.settings import DATA_FOLDER, MODELS_FOLDER_2

from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()
    task_name = args.task_name

    models_subdirectories = [x[0] for x in os.walk(MODELS_FOLDER_2)]
    models_subdirectories = sorted(models_subdirectories)



    data = list()
    for subdirectory in models_subdirectories:
        if task_name in subdirectory and 'MLP-LaBSE' in subdirectory:
            print(subdirectory)
            data_dict = gather_results(subdirectory, task_name)
            data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'MLP-LaBSE-' + task_name + '.csv'), index=False)


def gather_results(ft_model_dir: str, task_name: str) -> Dict[str, Any]:
    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(ft_model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'model.chkpt'))
    data['model_size'] = model_size

    for dirpath, dirnames, filenames in os.walk(os.path.join(MODELS_FOLDER_2, 'LaBSE')):
        for filename in [f for f in filenames if f.endswith(".bin")]:
            data['model_size'] += os.path.getsize(os.path.join(dirpath, filename))

    num_labels = get_num_labels(task_name)

    model = MLPClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(ft_model_dir, 'model.chkpt'),
        input_size=training_data_dict['input_size'],
        hidden_size=training_data_dict['hidden_size'],
        output_size=num_labels,
        learning_rate=training_data_dict['learning_rate'],
        weight_decay=training_data_dict['weight_decay'],
        dropout=training_data_dict['dropout']
    )
    labse_model = SentenceTransformer("sentence-transformers/LaBSE",
                                      cache_folder=os.path.join(MODELS_FOLDER_2, 'LaBSE'))

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labse_model.to(device)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    memory_params += sum([param.nelement() * param.element_size() for param in labse_model.parameters()])
    memory_buffers += sum([buf.nelement() * buf.element_size() for buf in labse_model.buffers()])

    memory_used = memory_params + memory_buffers  # in bytes

    data['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    for n, p in labse_model.named_parameters():
        parameters_num += p.nelement()

    data['parameters'] = parameters_num
    data['name'] = os.path.basename(ft_model_dir)
    return data


if __name__ == '__main__':
    main()
