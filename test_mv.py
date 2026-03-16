import torch
import sys
import os

def check_system():
    print("--- System Diagnostic ---")
    print("The goal of this file is to ensure that the environment is correctly set up for GPU usage with PyTorch.")
    print(f"Python Version : {sys.version}")
    print(f"Hostname         : {os.uname().nodename}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available : {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"Number of GPUs  : {device_count}")
        print(f"ID of used GPU : {current_device}")
        print(f"GPU Model  : {device_name}")
        
        # Test de calcul réel sur GPU
        x = torch.rand(5, 3).cuda()
        print("\nComputation Test : Tensor created successfully on the GPU.")
        print(f"Device of the tensor : {x.device}")
    else:
        print("\nERROR : PyTorch does not see any GPU.")
        print("Verify that you have properly loaded the CUDA module and that you are on a GPU node.")

if __name__ == "__main__":
    check_system()