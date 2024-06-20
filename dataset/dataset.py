from torch.utils.data import Dataset
import os
from config import Config
from random import sample
from torchvision import transforms as T
from PIL import Image
import numpy as np
class Datas(Dataset):
    def __init__(self, root, mod='train', sam='all', num=[]):
        self.img = []
        self.labels = []
        self.opt = Config
        if mod == 'train':
            labels_train = [i for i in range(len(self.opt.train_way))]
            if sam == 'all':
                for i in self.opt.train_way:
                    length = 0
                    mod_path = os.path.join(root, os.listdir(root)[i])
                    for img in os.listdir(mod_path):
                        self.img.append(os.path.join(mod_path, img))
                        self.labels.append(labels_train[self.opt.train_way.index(i)])
                        length += 1
                        if length > 200:
                            break
            else:
                labels = [i for i in range(len(num))]
                for i in num:
                    mod_path = os.path.join(root, os.listdir(root)[i])
                    imgs = sample(os.listdir(mod_path), self.opt.shot)
                    if sam == 'support':
                        for img_support in imgs:
                            self.img.append(os.path.join(mod_path, img_support))
                            self.labels.append(labels[num.index(i)])
                    elif sam == 'query':
                        length = 0
                        # for img_test in [x for x in os.listdir(mod_path) if x not in imgs]:
                        for img_test in os.listdir(mod_path):
                            self.img.append(os.path.join(mod_path, img_test))
                            self.labels.append(labels[num.index(i)])
                            length += 1
                            if length >= 200:
                                break

                    else:
                        print('支持集查询集选择错误！')
        elif mod == 'test':
            labels = [i for i in range(len(self.opt.test_way))]
            for i in self.opt.test_way:
                mod_path = os.path.join(root, os.listdir(root)[i])
                imgs = sample(os.listdir(mod_path), self.opt.shot)
                if sam == 'support':
                    for img_support in imgs:
                        self.img.append(os.path.join(mod_path,img_support))
                        self.labels.append(labels[self.opt.test_way.index(i)])
                elif sam == 'query':
                    for img_test in [x for x in os.listdir(mod_path) if x not in imgs]:
                        self.img.append(os.path.join(mod_path,img_test))
                        self.labels.append(labels[self.opt.test_way.index(i)])
                elif sam == 'all':
                    for img in os.listdir(mod_path):
                        self.img.append(os.path.join(mod_path, img))
                        self.labels.append(labels[self.opt.test_way.index(i)])
                else:
                    print("读取数据模式错误！应该输入'support'、'query'、'train'或者'test'")
        self.transform = T.Compose(
            T.ToTensor(),)

        # def _transform(n_px):
        #     return Compose([
        #         Resize(n_px, interpolation=BICUBIC),
        #         CenterCrop(n_px),
        #         _convert_image_to_rgb,
        #         ToTensor(),
        #         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        #     ])
    def __getitem__(self, item):
        data = self.img[item]
        # data = Image.open(self.img[item])
        # a = len(img.split())
        # data = np.array(Image.open(self.img[item]))
        label = self.labels[item]
        return data, label

    def __len__(self):
        return len(self.img)
