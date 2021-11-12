import gzip
import os
import shutil

import requests
from fasttext import load_model
from tqdm.auto import tqdm

from src.settings import MODELS_FOLDER

EN_MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
PL_MODEL_URL = 'https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin'

fasttext_folder = os.path.join(MODELS_FOLDER, 'fasttext')
os.makedirs(fasttext_folder, exist_ok=True)


def download_file(url: str, file_path: str):
    print(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")


def bin_to_vec(bin_model_path: str):
    model_bin_file = os.path.basename(bin_model_path)
    model_name = '.'.join(model_bin_file.split('.')[:-1])
    model_vec_file = model_name + '.vec'
    model_vec_path = os.path.join(os.path.dirname(bin_model_path), model_vec_file)

    print(f"Fastext {model_bin_file} -> {model_vec_file}")
    f = load_model(bin_model_path)
    words = f.get_words()

    with open(model_vec_path, 'w') as file_out:
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + '\n')
        for w in tqdm(words):
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass


en_output_gz = os.path.join(fasttext_folder, 'cc.en.300.bin.gz')
en_output = os.path.join(fasttext_folder, 'cc.en.300.bin')
download_file(EN_MODEL_URL, en_output_gz)

with gzip.open(en_output_gz, 'rb') as f_in:
    with open(en_output, 'wb') as f_out:
        tqdm(shutil.copyfileobj(f_in, f_out))

os.remove(en_output_gz)
bin_to_vec(en_output)

pl_output_bin = os.path.join(fasttext_folder, 'kgr10.plain.skipgram.dim300.neg10.bin')
download_file(PL_MODEL_URL, pl_output_bin)
bin_to_vec(pl_output_bin)
