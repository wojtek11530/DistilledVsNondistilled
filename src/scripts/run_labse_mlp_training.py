import argparse
import logging
import sys

from src.lightning_models.training import train_model

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
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--input_size",
                        default=768,
                        type=int,
                        help="The dimension of input (i.e. word embeddings dimension)")
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int,
                        help="The dimension of hidden layer")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Probability of dropout")
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument('--do_test',
                        action='store_true',
                        help='Performing evaluation on test set after training')

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    train_model(args)


if __name__ == '__main__':
    main()
