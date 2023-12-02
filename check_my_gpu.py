import torch
import os


def check_my_gpu():
    """Check if system has GPU available and print possible GPU details."""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print("\nGPU found!")
        number_of_gpus = torch.cuda.device_count()
        print("Number of GPUs: {}".format(torch.cuda.device_count()))
        print("GPU types:")
        for i in range(number_of_gpus):
            print("GPU {} type: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("No GPUs available!")


def check_my_cpu():
    """Check number of CPUs in partition and how many you are booking."""
    print("\nTotal number of CPU cores found: {}".format(os.cpu_count()))
    print("You are booking {} CPU cores.".format(len(os.sched_getaffinity(0))))


def main():
    check_my_cpu()
    check_my_gpu()


if __name__ == "__main__":
    main()
