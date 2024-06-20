from config import Config
from model.models import models
import torch
import random
from dataset.dataset import Datas
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *

opt = Config()
#
models = models(opt.CLIP_feature).to(opt.device)
if opt.new_parameter is False:
    models.load_state_dict(torch.load(opt.clip_model_path))
# models = torch.load('D:\DeepLearning\CLIP_Adapter\checkpoints\85.04%\clip_adapter.pt')
for name,param in models.named_parameters():
    if param.requires_grad is True:
        print(name,param.size())
#
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, models.parameters()),lr=opt.lr)
# task_remain = False
for epoth in range(opt.train_epoch):
    #支持集训练
    # if task_remain is False:
    # num = random.randint(3,7)
    sample = random.sample(range(0,len(opt.train_way)),len(opt.test_way))
    sample = [opt.train_way[i] for i in sample]
    text = [opt.text[i] for i in sample]
    feature_all = torch.zeros((len(opt.test_way), opt.CLIP_feature)).to(opt.device)
    if opt.shot != 0:
        support_data = Datas(opt.path_train_data,mod='train',sam='support',num=sample)
        support_dataloader = DataLoader(support_data,batch_size=opt.batch_size,shuffle=True)
        feature = torch.zeros((len(opt.test_way), opt.CLIP_feature)).to(opt.device)
        for j in range(opt.supoort_epoch):
            for _,(image,label) in enumerate(support_dataloader):
                image_feature,text_feature = models(image,text)
                label = label.to(opt.device)
                logic1 = torch.mm(image_feature,text_feature.transpose(0,1))
                loss1 = F.cross_entropy(logic1,label)
                # logic2 = calculate_distance(feature1=image_feature,feature2=feature_all,mod='cosin')
                # loss2 = F.cross_entropy(logic2,label)
                # loss = loss1 + loss2
                loss = loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for m in range(len(label)):
                    feature[label[m]] += image_feature[m]    #要求：batch_size大于test_way * test_shot
                feature = feature / opt.shot
        feature_all = feature.detach()
    # else:

    #查询集训练
    optimizer.zero_grad()
    query_data = Datas(opt.path_train_data,mod='train',sam='query',num=sample)
    # query_data = Datas(opt.path_train_data, mod='train', sam='query', num=sample)
    query_dataloader = DataLoader(query_data,batch_size=opt.batch_size,shuffle=True)
    # text = [opt.text[i] for i in opt.train_way]
    correct,length,total_loss= 0,0,0
    correct_text,correct_dis = 0,0
    for i,(image,label) in enumerate(query_dataloader):
        image_feature,text_feature = models(image,text)
        label = label.to(opt.device)
        if opt.shot == 0:
            feature_all = text_feature
        # feature_all.require_grad = False
        logic1 = torch.mm(image_feature,text_feature.transpose(0,1))
        # pred1 = logic1.max(1,keepdim=True)[1].squeeze()
        loss1 = F.cross_entropy(logic1,label)
        # correct_text += pred1.eq(label).sum().item()

        logic2 = calculate_distance(image_feature, feature_all, mod='emd').to(opt.device)
        # logic2 = torch.mm(image_feature,feature_all.transpose(0,1))
        # pred2 = logic2.max(1,keepdim=True)[1].squeeze()
        loss2 = F.cross_entropy(logic2, label)
        # correct_dis += pred2.eq(label).sum().item()

        # logic = logic1
        # logic = torch.mm(image_feature,text_feature.transpose(0,1))
        logic = F.softmax(logic1, dim=-1) + F.softmax(logic2, dim=-1)
        pred = logic.max(1,keepdim=True)[1].squeeze()
        correct += pred.eq(label).sum().item()

        length += len(label)

        # loss = F.cross_entropy(logic,label)
        # loss = loss1 + loss2
        loss = loss1
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    optimizer.zero_grad()
    acc = correct / length * 100
    acc_text = correct_text / length * 100
    acc_dis = correct_dis / length * 100
    # print('epoch:{:3d}   acc:{:.2f}   loss{:.4f}  acc_text:{:.2f}   acc_dis:{:.2f}  sample:{}'.format(epoth,acc,total_loss,acc_text,acc_dis,sample))
    print('epoch:{:3d}   acc:{:.2f}   loss{:.4f}  sample:{}'.format(epoth, acc,total_loss,sample))
    # if (epoth + 1) % 10 == 0:
    #     torch.save(models.state_dict(), opt.clip_model_path)
    # if acc < 90:
    #     task_remain = True
    # else:
    #     task_remain = False
torch.save(models.state_dict(), opt.clip_model_path)



