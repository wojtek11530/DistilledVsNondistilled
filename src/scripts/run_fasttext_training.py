import argparse
import logging
import sys

from src.fasttext_models.training import train_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="The name of fasttext model with pretrained word vectors stored in models/fasttext folder")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--dim",
                        default=300,
                        type=int,
                        help="The mdimmension of fasttext vectors")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    train_model(model_name=args.model_name, task_name=args.task_name, data_dir=args.data_dir, dim=args.dim)


if __name__ == '__main__':
    main()
