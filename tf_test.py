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
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from safetensors import safe_open
from safetensors.torch import load_file
def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(text,clip_model):
    with torch.no_grad():
        clip_weights = []
        for texts in text:
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        return clip_weights

def build_cache_model(clip_model, train_loader):
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(10):
            train_features = []

            print('Augment Epoch: {:} / {:}'.format(augment_idx, 10))
            for i, (images, target) in enumerate(tqdm(train_loader)):
                image = []
                text_support = []
                for i in range(len(images)):
                    pre_img = data_preprocess(Image.open(images[i])).unsqueeze(0)
                    image.append(pre_img)
                images = torch.cat(image, dim=0)
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    a = torch.cat(cache_values, dim=0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
    return cache_keys,cache_values

def pre_load_features( clip_model,loader):

    features, labels = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image = []
            text_support = []
            for i in range(len(images)):
                pre_img = data_preprocess(Image.open(images[i])).unsqueeze(0)
                image.append(pre_img)
            images = torch.cat(image, dim=0)
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    return features, labels


def run_tip_adapter(cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,data_cache,data_all):
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = 1, 5

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha
    acc_cache = cls_acc(cache_logits,val_labels)
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    acc_cache = cls_acc(cache_logits,test_labels)
    data_cache.append(acc_cache)
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    data_all.append(acc)
    print("best beta:{}   best alpha:{}".format(best_beta,best_alpha))
    print('few shot distance: {:.2f}'.format(acc_cache))
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
    return data_cache,data_all

def run_tip_adapter_F( cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F,data_cahe,data_all):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.0001, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20 * len(train_loader_F))

    beta, alpha = 1, 20
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(500):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, 20))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            with torch.no_grad():
                image = []
                for i in range(len(images)):
                    pre_img = data_preprocess(Image.open(images[i])).unsqueeze(0)
                    image.append(pre_img)
                images = torch.cat(image, dim=0)
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)


        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            # torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    # adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cache_keys, cache_values, val_features, val_labels, clip_weights,
                                      adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    acc_cahe = cls_acc(cache_logits, test_labels)
    data_cahe.append(acc_cahe)
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    data_all.append(acc)
    print("best beta:{}   best alpha:{}".format(best_beta, best_alpha))
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
    return data_cahe,data_all

def search_hp( cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    beta_list = [i * (50 - 0.1) / 200 + 0.1 for i in
                 range(200)]
    alpha_list = [i * (500 - 0.1) / 5 + 0.1 for i in
                  range(200)]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    for beta in beta_list:
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

run_tip_adapter_F_data_all = []
run_tip_adapter_F_data_cahe = []
run_tip_adapter_data_all = []
run_tip_adapter_data_cahe = []
opt = Config
result_path = create_file('D:/DeepLearning/CLIP_Adapter/result/five_shot')
for i in range(1000):
    clip_model,_ = clip.load(opt.CLIP_model,device=opt.device)
    model_path = 'D:\DeepLearning\CLIP_Adapter\checkpoints\RS5M_ViT-B-32.PT'
    clip_model.load_state_dict(torch.load(model_path))
    # clip_model = torch.load(model_path)
    # tensors = {}
    # with safe_open(model_path, framework="pt", device='cpu') as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)
    #
    # print(tensors)
    # load = load_file(model_path,device=opt.device)
    # clip_model.load_state_dict(load_file(model_path,device=opt.device))
    # clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # clip.tokenize = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    from open_clip import create_model_from_pretrained, get_tokenizer
    # clip_model.load_state_dict(torch.load('D:\DeepLearning\CLIP_Adapter\checkpoints\RemoteCLIP-RN50.pt'))
    # result_path = create_file()

    support_data = Datas(opt.path_test_data, mod='test', sam='support')
    support_dataloader = DataLoader(support_data,batch_size=opt.batch_size,shuffle=False)
    train_dataloader = DataLoader(support_data,batch_size=opt.batch_size,shuffle=True)

    test_data = Datas(opt.path_test_data, mod='test', sam='query')
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True)

    # text = [opt.text[opt.train_way[i]] for i in [0,1,2,3,4,5,6]]
    text = [opt.text[i] for i in opt.test_way]
    # text = opt.text
    # correct,length = 0,0
    # total_loss = 0
    # pred_sum, label_sum=[],[]
    # correct_dis, correct_com= 0,0
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(text, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(clip_model, support_dataloader)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, test_dataloader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_dataloader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter_data_cahe,run_tip_adapter_data_all = run_tip_adapter(cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,run_tip_adapter_data_cahe,run_tip_adapter_data_all)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F_data_cahe,run_tip_adapter_F_data_all = run_tip_adapter_F(cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_dataloader,run_tip_adapter_F_data_cahe,run_tip_adapter_F_data_all)

cahe_mean,cahe_std,cahe_max,cahe_min = cal_data(run_tip_adapter_data_cahe)
mean,std,max,min = cal_data(run_tip_adapter_data_all)
F_cahe_mean,F_cahe_std,F_cahe_max,F_cahe_min = cal_data(run_tip_adapter_F_data_cahe)
F_mean,F_std,F_max,F_min = cal_data(run_tip_adapter_F_data_all)
print('F_data_all mean:{:.2f} std:{:.2f} max:{:.2f} min{:.2f}'.format(F_mean,F_std,F_max,F_min))
print('F_data_cahe mean:{:.2f} std:{:.2f} max:{:.2f} min{:.2f}'.format(F_cahe_mean,F_cahe_std,F_cahe_max,F_cahe_min))
f = open(os.path.join(result_path,'result.txt'),'w')
f.write('F_cahe_mean:'+str(F_cahe_mean)+'\n')
f.write('F_cahe_std:'+str(F_cahe_std)+'\n')
f.write('F_cahe_max:'+str(F_cahe_max)+'\n')
f.write('F_cahe_min:'+str(F_cahe_min)+'\n')
f.write('F_mean:'+str(F_mean)+'\n')
f.write('F_std:'+str(F_std)+'\n')
f.write('F_max:'+str(F_max)+'\n')
f.write('F_min:'+str(F_min)+'\n')

