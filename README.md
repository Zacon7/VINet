

> This is a non-official PyTorch implementation of VINet[1] forked from [https://github.com/HTLife/VINet](https://github.com/HTLife/VINet). Aim of this repository is to update version written by HTLife and improve the documentation. Project is under construction.


# Memo:

- VINet uses FlownetS (simple) instead of FlownetC (correlation)
- Update later to RAFT if there is time: [https://arxiv.org/pdf/2003.12039.pdf](https://arxiv.org/pdf/2003.12039.pdf) [https://github.com/pytorch/vision/tree/main/references/optical_flow](https://github.com/pytorch/vision/tree/main/references/optical_flow)

# Installation

First install pip requirements

```
pip3 install -r requirements.txt
```

Then install PySophus

```
# Download and install PySophus
git clone https://github.com/arntanguy/PySophus.git
cd PySophus
git submodule init
git submodule update
python3 setup.py build_ext --inplace


# Add PySophus to bashrc or to Python path
export PYTHONPATH="$PYTHONPATH:<path to PySophus>"

example: export PYTHONPATH="$PYTHONPATH:/Desktop/Fusion-Project/PySophus"
```

# Download EuRoC MAV Dataset

```
# Create folder where to read data
cd ..
mkdir data
cd data
```

Save and unzip datafiles to this folder

[EuRoC MAV Dataset Homepage](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#available_data)

[EuRoC MAV Dataset download page](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/)


# Download pretrained Flownet model

Download trained model to same location as where you have the VINNet main.py file

[Download trained Flownet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)

[Link to Nvidia Flownet repository](https://github.com/NVIDIA/flownet2-pytorch)

# Training (update training process!!)
Log into container
```bash
sudo docker exec -it vinet bash
cd /notebooks/vinet
```

Execute main.py by
```bash
python3 main.py
```

# Note
## Network detail structure
![](./doc_fig/vinet.png)

![](./doc_fig/se3_def.png)



[1] Clark, Ronald, et al. "VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem." AAAI. 2017.
