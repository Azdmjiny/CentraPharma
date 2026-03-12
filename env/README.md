### 请先确保下载好miniconda
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`
`source ~/miniconda3/bin/activate`
`conda init`


### conda环境搭建
`conda env create -f environment.yml -n agent`
`conda env create -f AmberTools25.yml -n amber`
注：如有缺失可用requirements.txt补充下载


`apt-get update`
`apt-get install -y libcairo2-dev pkg-config python3-dev libxml2-dev libxslt1-dev`

`conda create -n pka310 python=3.10 -y`
`conda activate pka310`

`conda install -c conda-forge rdkit pandas numpy scipy svgutils cairosvg tqdm matplotlib pillow lxml -y`

`pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu`

`pip install torch-scatter==2.1.1+pt113cpu torch-sparse==0.6.17+pt113cpu -f https://data.pyg.org/whl/torch-1.13.1+cpu.html`
`pip install torch-geometric==2.0.1`

`git clone https://github.com/mayrf/pkasolver.git`
`cd pkasolver`
`python setup.py install`

### 补充包
`apt-get update`

`apt-get install -y libxrender1`

### ollama
`curl -fsSL https://ollama.com/install.sh | sh`

`ollama serve`

此处新开一个bash

`ollama pull qwen3:8b`