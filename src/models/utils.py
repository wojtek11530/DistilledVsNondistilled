import logging
import os

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def result_to_file(result: dict, file_name: str, verbose: bool = True) -> None:
    with open(file_name, "a") as writer:
        if verbose:
            logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            if verbose:
                logger.info(" %s = %s", key, str(result[key]))
            writer.write("%s = %s" % (key, str(result[key])))

        writer.write("")


def is_folder_empty(folder_name: str):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False
