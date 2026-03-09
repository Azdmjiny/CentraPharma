### 请先确保下载好miniconda

### conda环境搭建
`conda env create -f environment.yml -n agent`

注：如有确实可用requirements.txt补充下载

### 补充包
`apt-get update`

`apt-get install -y libxrender1`

### ollama
`curl -fsSL https://ollama.com/install.sh | sh`

`ollama serve`

此处新开一个bash

`ollama pull qwen3:8b`