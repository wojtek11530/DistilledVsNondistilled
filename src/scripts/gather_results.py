import argparse
import os

from src.settings import MODELS_FOLDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()

    task_name = args.task_name

    models_folders = get_immediate_subdirectories(MODELS_FOLDER)
    for model_folder in models_folders:
        task_models_directories = get_immediate_subdirectories(model_folder)
        for task_models_directory in task_models_directories:
            if task_name in task_models_directory:
                gather_results(task_models_directory)


def get_immediate_subdirectories(dir: str):
    return [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def gather_results(task_models_directory):
    print(task_models_directory)


if __name__ == '__main__':
    main()
