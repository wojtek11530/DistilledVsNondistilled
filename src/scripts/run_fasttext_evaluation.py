import argparse
import logging
import sys

from src.fasttext_models.evaluation import test_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        help="The direction of the trained fasttext model, i.e. folder where 'model.bin' file is stored")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to evaluate.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    test_model(model_dir=args.model_dir, task_name=args.task_name, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
