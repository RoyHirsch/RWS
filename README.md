# Random Walks for Temporal Action Segmentation with Timestamp Supervision

This repository contains the codebase and the experiments documentation for the paper: 
'Random Walks for Temporal Action Segmentation with Timestamp Supervision'

## Data
The project uses pre-computed I3D features for GTEA, 50Salads and Breakfast datasets. The features can be downloaded from [here](https://zenodo.org/record/3625992#.Y3VW_uxBxUd) (~30GB).
Please download the data to `./data` folder. The timestamps can be downloaded from [here](https://github.com/ZheLi2020/TimestampActionSeg/tree/main/data) and each `.npy` file should be extracted to the appropriate data folder in `./data`.

## Stand-alone Random Walk for Pseudo-Label Generation
Using the proposed random walk-based method for generating dense pseudo-labels from timestamps.
```
python ./stand_alone/stand_alone_main.py --dataset_name=50salads
```

## Train RWS model (Random Walk Segmentation model)
Here are instructions for training parametric action segmentation model using the three random-walk use-cases presented in the paper. The main script is `./rws/main.py`. Sample configuration files can be founf in `./rws/configs`. Configurations defined in `config.py` file and can be easily modified.

Example for running an experiment over GTEA dataset:
```
python ./rws/rws_main.py --config=./rws/configs/gtea_rw.yaml
```
