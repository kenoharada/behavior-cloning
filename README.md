# behavior-cloning

## SSH

```bash
# ssh -A -L 5900:localhost:5901 -L 8888:localhost:8889 {server_name}
ssh -A -o ServerAliveInterval=1 abci
```

## Setup bc env

Check your machine's CUDA version and set version in Dockerfile and bc_env.def  
https://catalog.ngc.nvidia.com/containers

```bash
git clone https://github.com/kenoharada/behavior-cloning.git
cd behavior-cloning
```

### Singularity

```bash
# export GROUP_ID={hoge}
# qrsh -g $GROUP_ID -l rt_AG.small=1 -l h_rt=1:00:00
# module load singularitypro
singularity build --fakeroot bc_env.sif bc_env.def
singularity shell --nv bc_env.sif
```

### Docker

```bash
docker build -t bc_env .
docker run --gpus all --rm -it --shm-size=48gb -p 5901:5900 -p 8889:8888 --mount type=bind,src=$PWD,dst=/root/bc_template  --name `whoami`_bc bc_env
```

## Install packages for experiments

```bash
python3 -m venv ~/venv/bc_env
source ~/venv/bc_env/bin/activate
pip3 install --upgrade pip
pip3 install jupyterlab wandb matplotlib
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

git clone https://github.com/facebookresearch/r3m.git
cd r3m
# edit pillow version
pip3 install -e .

cd ..
pip3 install -e .
# Check jupyter lab in lab remote server
# nohup jupyter lab --port 8888 --ip=0.0.0.0 --allow-root >> jupyer.log &
## acess with token via your local machine's browser
# cat jupyer.log | grep 127.0.0.1:8888 | tail -n 1
# Check jupyter lab in ABCI
# https://github.com/kenoharada/abci_tutorial/blob/main/jupyter_tutorial.md
# https://docs.abci.ai/ja/tips/jupyter-notebook/
# check training
cd experiments
CUDA_VISIBLE_DEVICES=0 nohup python train.py --epochs 30 >> train.log &
```

## batch job in ABCI

```bash
qsub -g $GROUP_ID job.sh
```
