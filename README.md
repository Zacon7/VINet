

> This is a non-official PyTorch implementation of VINet[1] forked from [https://github.com/HTLife/VINet](https://github.com/HTLife/VINet). Aim of this repository is to update version written by HTLife and improve the documentation. Project are still under construction.

# Installation

It's recommand to use docker image to run this project.
[Docker image installation guide](https://github.com/HTLife/VINet/wiki/Installation-Guide)

# Training
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
