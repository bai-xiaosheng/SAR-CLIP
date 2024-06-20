from config import Config
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as F
def calculate_distance(feature1,feature2,mod='cosin'):
    if mod == 'cosin':
        logic = torch.mm(feature1,feature2.transpose(0,1))
    elif mod == 'emd':
        weight1,weight2= get_weight_vector(feature1,feature2)
        logic = []
        for i in range(feature1.size(0)):
            for j in range(feature2.size(0)):
                logic.append(-wasserstein_distance(feature1[i].cpu().detach().numpy(),feature2[j].cpu().detach().numpy()))
        logic = torch.Tensor(logic).resize(feature1.size(0),feature2.size(0))
    else:
        print('请输入正确的度量方式，并且保证输入特征维度相同！')
    return logic


def get_weight_vector(feature1,feature2):  #60,1024  3,1024
    M = feature1.size(0)
    N = feature2.size(0)

    feature1 = feature1.unsqueeze(1).repeat(1,N,1)
    feature2 = feature2.unsqueeze(0).repeat(M,1,1)

    mean1 = torch.mean(feature1,dim=-1).unsqueeze(2)
    mean2 = torch.mean(feature2,dim=-1).unsqueeze(2)

    combination = feature1 * feature2
    combination = combination.view(M, N, -1)
    weight1 = combination / mean2
    weight2 = combination / mean1
    weight1 = F.relu(weight1) + 1e-3
    weight2 = F.relu(weight2) + 1e-3
    return weight1, weight2


# def get_weight_vector(feature1, feature2):  #节省空间,输入是（1024） （1024）
#     mean1 = torch.mean(feature1)
#     mean2 = torch.mean(feature2)
#     combination = feature1 * feature2
#     combination = F.relu(combination) + 1e-3
#     weight1 = combination / mean2
#     weight2 = combination / mean1
#     return weight1, weight2