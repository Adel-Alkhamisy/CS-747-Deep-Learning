python3 -m venv my_venv
source my_venv/bin/activate
pip install --upgrade pip
pip install torchtext portalocker
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
pip install ipykernel
python -m ipykernel install --user --name=my_venv
pip install Unidecode
pip uninstall torch
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torch --upgrade
pip install fastai==1.0.61