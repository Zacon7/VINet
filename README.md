

> This is a non-official PyTorch implementation of VINet[1] forked from [https://github.com/HTLife/VINet](https://github.com/HTLife/VINet). Aim of this repository is to update version written by HTLife and improve the documentation. Project is under construction.


# Memo:

- VINet uses FlownetS (simple) instead of FlownetC (correlation)
- Update later to RAFT if there is time: [https://arxiv.org/pdf/2003.12039.pdf](https://arxiv.org/pdf/2003.12039.pdf) [https://github.com/pytorch/vision/tree/main/references/optical_flow](https://github.com/pytorch/vision/tree/main/references/optical_flow)

# Installation

First install pip requirements

```
pip install -r requirements.txt
```

Then install PySophus

```
git clone https://github.com/arntanguy/PySophus.git
cd PySophus
git submodule init
git submodule update
python setup.py build_ext --inplace
# Add PySophus to bashrc
export PYTHONPATH="$PYTHONPATH:<path to PySophus>"
```

# Download EuRoC MAV Dataset

[EuRoC MAV Dataset Homepage](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#available_data)


[EuRoC MAV Dataset download page](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/)

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
