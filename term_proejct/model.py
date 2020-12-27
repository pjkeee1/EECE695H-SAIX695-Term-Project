import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.utils import square_euclidean_metric
import math


'''
baseline
'''

def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size= 2, stride= 2),
        nn.ReLU(),
    )

# nway= 5, h= 400, w= 400

class FewShotModel(nn.Module):
    # def __init__(self):
    #     super(FewShotModel, self).__init__()
    #     self.features = nn.Sequential(
    #         conv_block(3, 64),
    #         conv_block(64, 128),
    #         conv_block(128, 256),  
    #         conv_block(256, 512),
    #         conv_block(512, 1024),
    #     )
    def __init__(self):
        super(FewShotModel, self).__init__()
        self.features = models.resnet50() # 마지막 train accuracy : 50%, validation accuracy : 34 %
        # self.features = models.alexnet() # 안됨
        # self.features = models.vgg16() # 안됨
        
        
        
    def forward(self, x):
        x = self.features(x)
        embedding_vector = x.view(x.size()[0], -1)
        
        # # power transform
        # beta = 0.5
        # e = 1e-6        
        # embedding_vector = torch.pow(embedding_vector+ e, beta)
        # embedding_vector = embedding_vector/embedding_vector.norm(dim= 0, keepdim=True)
        
        return embedding_vector
    
    
class Distance_metric(nn.Module):
    def __init__(self):
        super(Distance_metric, self).__init__()
        
    def forward(self, embd_s, embd_q):
        prototype = torch.mean(embd_s, 1) # size : (nway, output dimension of embedding)
        logits = square_euclidean_metric(embd_q, prototype)
        
        return logits
            
       
    
class Prototypical_Network(nn.Module):       
    def __init__(self):
        super(Prototypical_Network, self).__init__()
        self.embd_g = FewShotModel()
        self.embd_f = FewShotModel()
        self.comp = Distance_metric()
        
    def forward(self, data_shot, data_query, nway, kshot, query):
        
        """
        produce the embedded support set
        """
        h = data_shot.size(2)
        w = data_shot.size(3)
        data_shot_reshape = torch.reshape(data_shot, [nway, kshot, 3, h, w])
        
        embed_support_set = []
        for i in range(nway):
            temp_embed = torch.squeeze(self.embd_g(data_shot_reshape[i, :, :, :, :]), 0)
            embed_support_set.append(temp_embed)
        embed_support_set = torch.stack(embed_support_set) # size : (nway, kshot, output dimension of embedding)
        
        embed_query_set = self.embd_f(data_query) # size : (query, output dimension of embedding)
        
        logits = self.comp(embed_support_set, embed_query_set) # size : (query, nway)
                      
                        
        return logits
        
        
        
'''
Learning to Compare: Relation Network for Few-Shot Learning
'''

'''
Reference
'''
'''
-------------------------------------
Project: Learning to Compare: Relation Network for Few-Shot Learning
Date: 2017.9.21
Author: Flood Sung
All Rights Reserved
----      
'''
  
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size= 3, padding=0),
                        nn.BatchNorm2d(64, momentum= 1, affine= True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size= 3, padding= 0),
                        nn.BatchNorm2d(64, momentum= 1, affine= True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
                        nn.BatchNorm2d(64, momentum= 1, affine= True),
                        nn.ReLU(),
                        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
                        nn.BatchNorm2d(64, momentum= 1, affine= True),
                        nn.ReLU(),
                        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size()[0],-1)
        return out # out.shape : torch.Size([B, 1600]) 
        
class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size= 3, padding= 1),
                        nn.BatchNorm2d(64, momentum= 1, affine= True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size= 3,padding= 1),
                        nn.BatchNorm2d(64, momentum= 1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out       
        
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
class main_network(nn.Module):
    def __init__(self):
        super(main_network, self).__init__()
        self.CNNEncoder = CNNEncoder().apply(weights_init)
        self.RelationNetwork = RelationNetwork(64, 8).apply(weights_init)        
        
    def forward(self, data_shot, data_query, nway, kshot, query):
        
        """
        produce the embedded support set
        """
        _, _, h, w = data_shot.shape
        data_shot_reshape = torch.reshape(data_shot, [nway, kshot, 3, h, w])
        
        embed_support_set = []
        for i in range(nway):
            temp_embed = torch.squeeze(self.CNNEncoder(data_shot_reshape[i, :, :, :, :]), 0) # size : (kshot, output dimension of embedding) or (kshotx64*5*5)
            temp_embed = torch.sum(temp_embed, 0) # size : (1, output dimension of embedding(kwhotx64*5*5))
            embed_support_set.append(temp_embed)
        embed_support_set = torch.stack(embed_support_set) # size : (nway, output dimension of embedding)
        embed_support_set = embed_support_set.view(nway, 64, 5, 5)
        
        embed_support_set_ext = embed_support_set.unsqueeze(0).repeat(query, 1, 1, 1, 1) # size : (query, nway, 64, 5, 5)
        
               
        embed_query_set = self.CNNEncoder(data_query) # size : (query, output dimension of embedding)
        embed_query_set = embed_query_set.view(query, 64, 5, 5)
        
        embed_query_set_ext = embed_query_set.unsqueeze(0).repeat(nway, 1, 1, 1, 1)
        embed_query_set_ext = torch.transpose(embed_query_set_ext, 0, 1) # size : (query, nway, 64, 5, 5)
        
        relation_pairs = torch.cat((embed_support_set_ext, embed_query_set_ext), 2).view(-1, 64*2, 5, 5)
        logits = self.RelationNetwork(relation_pairs).view(-1, nway)
        
        
        return logits
        
        
# class CNNEncoder(nn.Module):
#     def __init__(self):
#         super(CNNEncoder, self).__init__()
#         self.features = models.resnet18()
        
#     def forward(self, x):
#         x = self.features(x)
#         embedding_vector = x.view(x.size()[0], -1)
        
        
#         return embedding_vector # out.shape : torch.size([B, 1000])
    
# class RelationNetwork(nn.Module):
#     def __init__(self):
#         super(RelationNetwork, self).__init__()
#         self.layer1 = nn.Sequential(
#                         nn.Conv2d(20, 10, kernel_size= 3, padding= 1),
#                         nn.BatchNorm2d(10, momentum= 1, affine= True),
#                         nn.ReLU(),
#                         nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#                         nn.Conv2d(10, 10, kernel_size= 3,padding= 1),
#                         nn.BatchNorm2d(10, momentum= 1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool2d(2))
#         self.fc1 = nn.Linear(40, 20)
#         self.fc2 = nn.Linear(20, 10)
#         self.fc3 = nn.Linear(10, 5)
#         self.fc4 = nn.Linear(5, 1)

#     def forward(self,x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0),-1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#         out = F.sigmoid(self.fc4(out))
#         return out      
        
# class main_network(nn.Module):
#     def __init__(self):
#         super(main_network, self).__init__()
#         self.CNNEncoder = CNNEncoder()
#         self.RelationNetwork = RelationNetwork()   
        
#     def forward(self, data_shot, data_query, nway, kshot, query):
        
#         """
#         produce the embedded support set
#         """
#         _, _, h, w = data_shot.shape
#         data_shot_reshape = torch.reshape(data_shot, [nway, kshot, 3, h, w])
        
#         embed_support_set = []
#         for i in range(nway):
#             temp_embed = torch.squeeze(self.CNNEncoder(data_shot_reshape[i, :, :, :, :]), 0) # size : (kshot, output dimension of embedding) or (kshotx64*5*5)
#             temp_embed = torch.sum(temp_embed, 0) # size : (1, output dimension of embedding(kwhotx64*5*5))
#             embed_support_set.append(temp_embed)
#         embed_support_set = torch.stack(embed_support_set) # size : (nway, output dimension of embedding)
#         embed_support_set = embed_support_set.view(nway, 10, 10, 10)
        
#         embed_support_set_ext = embed_support_set.unsqueeze(0).repeat(query, 1, 1, 1, 1) # size : (query, nway, 64, 5, 5)
        
               
#         embed_query_set = self.CNNEncoder(data_query) # size : (query, output dimension of embedding)
#         embed_query_set = embed_query_set.view(query, 10, 10, 10)
        
#         embed_query_set_ext = embed_query_set.unsqueeze(0).repeat(nway, 1, 1, 1, 1)
#         embed_query_set_ext = torch.transpose(embed_query_set_ext, 0, 1) # size : (query, nway, 64, 5, 5)
        
#         relation_pairs = torch.cat((embed_support_set_ext, embed_query_set_ext), 2).view(-1, 10*2, 10, 10)
#         logits = self.RelationNetwork(relation_pairs).view(-1, nway)
        
        
#         return logits        
        
        
        
        
        
        
        
        