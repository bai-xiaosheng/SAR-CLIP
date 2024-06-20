import clip
from config import Config
import torch
import torch.nn as nn
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image
class models(nn.Module):
    def __init__(self,c_in=1024):
        super(models,self).__init__()
        self.opt = Config
        #读取clip模型
        # if self.opt.new_parameter is True:
        self.clip,_ = clip.load(self.opt.CLIP_model,device=self.opt.device)
        # 对clip的位置编码进行微调
        for name, parameter in self.clip.named_parameters():
            # if name != 'visual.attnpool.positional_embedding':
            parameter.requires_grad = False
        # else:
        #     self.clip = torch.load(self.opt.clip_model_path)
        #定义编码器
        # self.fc_image = nn.Sequential(
        #     nn.Linear(c_in,c_in),
        #     nn.ReLU(),
        #     nn.Linear(c_in, c_in),
        #     nn.ReLU()
        # )
        # self.fc_text = nn.Sequential(
        #     nn.Linear(c_in,c_in),
        #     nn.ReLU(),
        #     nn.Linear(c_in, c_in),
        #     nn.ReLU()
        # )

    def forward(self,image,text):
        #预处理数据
        image = self.preprocess(image)
        text = clip.tokenize(text).to(self.opt.device)
        #clip编码
        # image_feature = self.clip.encode_image(image).to(torch.float32)
        # text_feature = self.clip.encode_text(text).to(torch.float32)
        image_feature = self.clip.encode_image(image)
        text_feature = self.clip.encode_text(text)
        # #Adapter结构
        # if self.opt.Image_Adapter is True:
        #     image_feature = self.Image_encoder(image_feature)
        # if self.opt.text_Adapter is True:
        #     text_feature = self.text_Adapter(text_feature)
        #归一化
        image_feature = image_feature / image_feature.norm(dim=1,keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=1,keepdim=True)
        return image_feature,text_feature

    #预处理图像
    def preprocess(self,image):
        length = len(image)
        transfer = Compose([
            CenterCrop((224,224)),
            self._convert_image_to_rgb,
            ToTensor(),
        ])
        images = []
        for i in range(length):
            image_pre = transfer(Image.open(image[i])).unsqueeze(0)
            images.append(image_pre)
        # a = image_pre.cpu().detach().numpy()[0,0:,:]
        images = torch.cat(images,dim=0).to(self.opt.device)
        return images
    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")

    #图像Adapter编码
    def Image_encoder(self,image):
        image_feature = self.fc_image(image)
        out = image_feature + image
        return out

    #文本Adapter编码
    def text_Adapter(self,text):
        text_feature = self.fc_text(text)
        out = text_feature + text
        return out