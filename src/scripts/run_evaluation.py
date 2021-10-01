import argparse
import logging
import sys

from src.models.evaluation import test_model
from src.models.training import train_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        help="The direction of the fine-tuned model.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    test_model(model_dir=args.model_dir, task_name=args.task_name, data_dir=args.data_dir,
               batch_size=args.batch_size, max_seq_length=args.max_seq_length)


if __name__ == '__main__':
    main()
