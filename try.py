from sgmse.data_module import SpecsDataModule, Specs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sgmse.visual_module import ResNet, visualTCN
import torch.nn as nn


frontend3D = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                        nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0))
                        )
resnet = ResNet(128)
tcn = visualTCN(64) 

data_dir = "/home3/hexin/avse_data/"
data_module = SpecsDataModule(base_dir=data_dir)
data_module.setup(stage="test")
dataset = data_module.test_dataloader()
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
clean, noisy, video = next(iter(train_dataloader))
print(f"input audio size: {clean.size()}")
print(f"input video size: {video.size()}")

if video.ndim!=4:
      B=1
      #print('inputbatch shape', inputBatch.shape)
      #T = inputBatch.shape[0]
      B, T, W, H = video.shape
      video = video.view(T, 1, 1, W, H)

      #inputBatch=np.expand_dims(inputBatch, 1)
      #inputBatch=np.expand_dims(inputBatch, 1)
      
else:
      B, T, W, H = video.shape
      video = video.view(B*T, 1, 1, W, H)
print(f"input video size: {video.size()}")
output = frontend3D(video)
print(f"after 3D front end: {output.size()}")
output = output.transpose(1,2)
print(f"after transpose: {output.size()}")
output = output.reshape(output.shape[0]*output.shape[1], output.shape[2], output.shape[3], output.shape[4])
print(f"after reshape: {output.size()}")
output = resnet(output)
print(f"after resnet: {output.size()}")
output = output.view(B, T, 128) 			
output = output.transpose(1,2) 
print(f"before TCN: {output.size()}")
output = tcn(output)
print(f"after TCN: {output.size()}")
output = output.transpose(1,2)
print(f"final: {output.size()}")