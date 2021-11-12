import argparse
import logging
import sys

from src.transformer_models.training import train_model

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
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="The name of pretrained model availiblie in HuggingFace library")
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
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Number of steps to perform linear learning rate warmup for.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    train_model(model_name=args.model_name, task_name=args.task_name, data_dir=args.data_dir,
                epochs=args.num_train_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, max_seq_length=args.max_seq_length)


if __name__ == '__main__':
    main()
