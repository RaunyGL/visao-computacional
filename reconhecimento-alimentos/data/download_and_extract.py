import urllib.request
import tarfile
import os

# URLs do dataset
url = 'https://data.vision.ee.ethz.ch/cvl/food-101/food-101.tar.gz'

# Caminho para salvar o dataset
dataset_path = 'data/food-101'

# Função para baixar e extrair o dataset
def download_and_extract(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(path, filename)
    
    # Baixar
    urllib.request.urlretrieve(url, filepath)
    
    # Extrair
    with tarfile.open(filepath, 'r:gz') as tar_ref:
        tar_ref.extractall(path)

download_and_extract(url, dataset_path)
