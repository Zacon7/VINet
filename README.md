

> This is a non-official PyTorch implementation of VINet[1] forked from [https://github.com/HTLife/VINet](https://github.com/HTLife/VINet). Aim of this repository is to update version written by HTLife and improve the documentation. Project is under construction.


# Memo:

- VINet uses FlownetS (simple) instead of FlownetC (correlation)
- Update later to RAFT if there is time: [https://arxiv.org/pdf/2003.12039.pdf](https://arxiv.org/pdf/2003.12039.pdf) [https://github.com/pytorch/vision/tree/main/references/optical_flow](https://github.com/pytorch/vision/tree/main/references/optical_flow)

# Installation

1. Clone this repository

```
git clone https://github.com/Saukkoriipi/VINet.git
```

2. Install all required Python packages with command:

```
pip3 install -r requirements.txt
```

3. Download wanted EuRoC MAV dataset to folder data

- [EuRoC MAV Dataset Homepage](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#available_data)

- [EuRoC MAV Dataset download page](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/)

4. Pre-process downloaded data to correct format

Add instructions!!

5. Download pretrained model

Add link!!

6. From main.py choose if you want to run VINet training, testing or inference

Add guide when ready!!

# Download pretrained Flownet model

Download trained model to same location as where you have the VINNet main.py file

[Download trained Flownet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)

[Link to Nvidia Flownet repository](https://github.com/NVIDIA/flownet2-pytorch)

# Note
## Network detail structure
![](./doc_fig/vinet.png)

![](./doc_fig/se3_def.png)



[1] Clark, Ronald, et al. "VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem." AAAI. 2017.