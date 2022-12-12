import torch
import torchvision
import models
from PIL import Image
import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


im_paths = ['./data/images/bikes.bmp', './data/images/bikes_gb.bmp', './data/images/bikes_wn.bmp', './data/images/building2.bmp', './data/images/building2_gb.bmp', './data/images/building2_wn.bmp', './data/images/lighthouse.bmp', './data/images/lighthouse_gb.bmp', './data/images/lighthouse_wn.bmp', './data/images/monarch.bmp', './data/images/monarch_gb.bmp', './data/images/monarch_wn.bmp', './data/images/sailing4.bmp', './data/images/sailing4_gb.bmp', './data/images/sailing4_wn.bmp', './data/images/statue.bmp', './data/images/statue_gb.bmp', './data/images/statue_wn.bmp', './data/images/stream.bmp', './data/images/stream_gb.bmp', './data/images/stream_wn.bmp', './data/images/studentsculpture.bmp', './data/images/studentsculpture_gb.bmp', './data/images/studentsculpture_wn.bmp']
for im_path in im_paths:
  model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
  model_hyper.train(False)
  # load our pre-trained model on the koniq-10k dataset
  model_hyper.load_state_dict((torch.load('./pretrained/koniq_pretrained.pkl')))

  transforms = torchvision.transforms.Compose([
                      torchvision.transforms.Resize((512, 384)),
                      torchvision.transforms.RandomCrop(size=224),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                       std=(0.229, 0.224, 0.225))])

  # random crop 10 patches and calculate mean quality score
  pred_scores = []
  for i in range(10):
      img = pil_loader(im_path)
      img = transforms(img)
      img = torch.tensor(img.cuda()).unsqueeze(0)
      paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

      # Building target network
      model_target = models.TargetNet(paras).cuda()
      for param in model_target.parameters():
          param.requires_grad = False

      # Quality prediction
      pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
      pred_scores.append(float(pred.item()))
  score = np.mean(pred_scores)
  # quality score ranges from 0-100, a higher score indicates a better quality
  print('Predicted quality score: %.2f' % score)
