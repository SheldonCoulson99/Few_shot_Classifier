#jupyter notebook
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
import torch
print(torch.__version__)
print(torch.cuda.is_available())