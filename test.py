from config import Config
# from model.models import models
import torch
from dataset.dataset import Datas
from torch.utils.data import DataLoader
from model import models
import torch.nn.functional as F
from utils import *
import clip
import os

opt = Config
# clip_model,_ = clip.load(opt.CLIP_model,device=opt.device)
# result_path = create_file()
if opt.shot == 0:
    model = models(opt.CLIP_feature).to(opt.device)
    # model = torch.load(opt.clip_model_path).to(opt.device)
    model.load_state_dict(torch.load(opt.clip_model_path))
    # model = torch.load('D:\DeepLearning\CLIP_Adapter\checkpoints\85.04%\clip_adapter.pt')
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(name, param.size())
    # query_data = Datas(opt.path_train_data,mod='train',sam='query',num=sample)
    # test_data = Datas(opt.path_train_data,mod='train',sam='query',num=[0,1,2,3,4,5,6])
    test_data = Datas(opt.path_test_data, mod='test', sam='all')
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True)
    # text = [opt.text[opt.train_way[i]] for i in [0,1,2,3,4,5,6]]
    text = [opt.text[i] for i in opt.test_way]
    correct,length = 0,0
    total_loss = 0
    pred_sum, label_sum=[],[]
    correct_dis, correct_com= 0,0
    for i,(image,label) in enumerate(test_dataloader):
        image_feature,text_feature = model(image,text)
        # text = clip.tokenize(text).to(opt.device)
        # text_feature2 = clip_model.encode_text(text)
        label = label.to(opt.device)
        logic = torch.mm(image_feature,text_feature.transpose(0,1))
        # logic2 = calculate_distance(image_feature,text_feature,mod='cosin').to(opt.device)
        # logic_com = F.softmax(logic2,dim=-1) + F.softmax(logic,dim=-1)
        pred,loss,correct = calculate_metric(logic,label,correct)
        # pred = logic.max(1,keepdim=True)[1].squeeze()
        # correct += pred.eq(label).sum().item()

        length += len(label)
        # loss = F.cross_entropy(logic, label)
        # _,loss2,correct_dis = calculate_metric(logic2,label,correct_dis)
        # _,loss_com,correct_com = calculate_metric(logic_com,label,correct_com)
        # pred = logic.max(1,keepdim=True)[1].squeeze()
        # correct += pred.eq(label).sum().item()
        # loss = F.cross_entropy(logic,label)
        pred_sum.append(pred)
        label_sum.append(label.to(opt.device))
        # correct += pred.eq(label.to(opt.device)).sum().item()
        # length += len(label)
        # loss = F.cross_entropy(logic,label.to(opt.device))
        total_loss += loss
    acc = correct / length * 100
    acc_dis = correct_dis / length * 100
    acc_com = correct_com / length * 100
    print('zero shot acc:{:.2f}   total_loss:{:.4f}  acc_dis:{:.2f}  acc_com:{:.2f}'.format(acc, total_loss,acc_dis,acc_com))
    print('zero shot acc:{:.2f}   total_loss:{:.4f}  correct:{}  length:{}'.format(acc, total_loss, correct,length))
    result_path = draw_confusion_matrix(label_sum,pred_sum,opt.result_path)
    data = [[opt.test_way,opt.shot,acc,'17',result_path]]
    write_excel(os.path.join(opt.result_path,'result.xlsx'),data)


if opt.shot !=0:
    max_acc = 0
    max_acc_text, max_acc_dis= 0, 0
    if opt.shot == 1:
        result_path = os.path.join(opt.root_path,'result/one_shot')
    elif opt.shot == 5:
        result_path = os.path.join(opt.root_path, 'result/five_shot')
    else:
        print('error')
    result_path = create_file(result_path)
    for epoch in range(opt.test_epoch):
        model = torch.load(opt.clip_model_path).to(opt.device)
        # model = models(opt.CLIP_feature).to(opt.device)
        # model = torch.load('D:\DeepLearning\CLIP_Adapter\checkpoints\85.04%\clip_adapter.pt')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=opt.lr)
        text = [opt.text[i] for i in opt.test_way]
        support_data = Datas(opt.path_test_data,mod='test',sam='support')
        support_dataloader = DataLoader(support_data,batch_size=opt.batch_size,shuffle=True)
        feature_all = torch.zeros((len(opt.test_way),opt.CLIP_feature)).to(opt.device)
        correct, length = 0, 0
        correct_text, correct_dis = 0, 0
        for i in range(opt.supoort_epoch):
            feature = torch.zeros((len(opt.test_way), opt.CLIP_feature)).to(opt.device)
            for _,(image,label) in enumerate(support_dataloader):
                image_feature,text_feature = model(image,text)
                label = label.to(opt.device)
                logic1 = torch.mm(image_feature,text_feature.transpose(0,1))
                logic2 = calculate_distance(image_feature,feature_all,mod='emd').to(opt.device)
                logic = F.softmax(logic1,dim=-1) + F.softmax(logic2,dim=-1)
                # logic = logic1
                loss1 = F.cross_entropy(logic1,label)
                loss2 = F.cross_entropy(logic2,label)

                loss = loss2 + loss1
                # loss = loss1
                _,_,correct = calculate_metric(logic,label,correct)
                _,_,correct_text = calculate_metric(logic1,label,correct_text)
                _,_,correct_dis = calculate_metric(logic2,label,correct_dis)
                length += len(label)
                optimizer.zero_grad()
                # loss1.backward(retain_graph=True)
                # loss2.backward()
                loss.backward()
                optimizer.step()
                for j in range(len(label)):
                    feature[label[j]] += image_feature[j]    #要求：batch_size大于test_way * test_shot
            feature_all = feature.detach() / opt.shot
        print('support: acc{:.2f}   acc_text{:.2f}   acc_dis{:.2f}'.format(correct/length*100,correct_text/length*100,correct_dis/length*100))
        #
        query_data = Datas(opt.path_test_data,mod='test',sam='all')
        query = DataLoader(query_data,batch_size=opt.batch_size,shuffle=True)
        correct,length = 0,0
        total_loss = 0
        pred_sum,label_sum=[],[]
        correct_text, correct_dis = 0, 0
        for i,(image,label) in enumerate(query):
            image_feature,text_feature = model(image,text)
            label = label.to(opt.device)
            logic1 = torch.mm(image_feature, text_feature.transpose(0, 1))
            logic2 = calculate_distance(image_feature, feature_all, mod='emd').to(opt.device)
            logic = F.softmax(logic1,dim=-1) + F.softmax(logic2,dim=-1)
            # logic = logic1
            _,_,correct_text = calculate_metric(logic1,label,correct_text)
            _,_,correct_dis = calculate_metric(logic2,label,correct_dis)
            pred = logic.max(1,keepdim=True)[1].squeeze()
            correct += pred.eq(label).sum().item()
            length += len(label)
            pred_sum.append(pred)
            label_sum.append(label)
        acc = correct  / length * 100
        acc_text = correct_text / length * 100
        acc_dis = correct_dis / length * 100
        max_acc = max(max_acc,acc)
        max_acc_text = max(max_acc_text,acc_text)
        max_acc_dis = max(max_acc_dis, acc_dis)
        print('epoch:{:3d}   acc:{:.2f}   acc_text{:.2f}   acc_dis{:.2f}'.format(epoch,acc,acc_text,acc_dis))
        draw_confusion_matrix( label_true=label_sum, label_pred=pred_sum,save_path=result_path)
    print('max_acc is {:.2f}   max_acc_text{:.2f}   max_acc_dis{:.2f}'.format(max_acc,max_acc_text,max_acc_dis))
    data = [[opt.test_way,opt.shot,'15',max_acc,result_path,max_acc_text,max_acc_dis]]
    write_excel(os.path.join(opt.result_path,'result.xlsx'),data)






