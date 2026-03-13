### Please make sure you've downloaded Miniconda

### conda environment
Create environment "agent"
```bash
conda env create -f environment.yml -n agent
```
Create environment "amber"
```bash
conda env create -f AmberTools25.yml -n amber
```
Create environment "pka310"
```bash
apt-get update
apt-get install -y libcairo2-dev pkg-config python3-dev libxml2-dev libxslt1-dev libxrender1

conda create -n pka310 python=3.10 -y
conda activate pka310

conda install -c conda-forge rdkit pandas numpy scipy svgutils cairosvg tqdm matplotlib pillow lxml -y

pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch-scatter==2.1.1+pt113cpu torch-sparse==0.6.17+pt113cpu -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install torch-geometric==2.0.1

git clone https://github.com/mayrf/pkasolver.git
cd pkasolver
python setup.py install
```
### ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```
run this in a new bash:
```bash
ollama pull qwen3:8b
```
