import os
import zipfile

import requests

from settings import DATA_FOLDER

url = 'https://clarin-pl.eu/dspace/bitstream/handle/11321/798/multiemo.zip?sequence=2&isAllowed=y'

output_zip = os.path.join(DATA_FOLDER, 'multiemo.zip')

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_zip, 'wb') as f:
        f.write(response.raw.read())

with zipfile.ZipFile(output_zip, "r") as zip_ref:
    zip_ref.extractall(DATA_FOLDER)

os.remove(output_zip)
os.remove(os.path.join(DATA_FOLDER, 'readme.txt'))
