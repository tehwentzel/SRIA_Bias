import torch
import pandas as pd
import numpy as np
from scipy.ndimage import rotate
import Utils
import cv2
from facenet_pytorch import InceptionResnetV1

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        return x
    
class BasicCnnEmbedding(torch.nn.Module):
               
    def __init__(self,
                 filter_sizes = [8,16,16,32],
                 filter_kernel_sizes = [32,16,8,4],
                 hidden_sizes = [500],
                 embedding_size = 256,
                 initial_dropout = .01,
                 linear_dropout = .2,
                 try_cuda=True,
                ):
        super().__init__()
        
        if try_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')
            
        
        self.init_dropout = torch.nn.Dropout(initial_dropout)
        
        
        convs = []
        curr_channels = 3
        for size,ksize in zip(filter_sizes,filter_kernel_sizes):
            layer = ConvBlock(curr_channels,size,ksize)
            curr_channels=size
            convs.append(layer)
        
        self.convs = torch.nn.ModuleList(convs)
        self.linear_dropout = torch.nn.Dropout(linear_dropout)
        
        self.flatten = torch.nn.Flatten()
        curr_size = 0
        fcs = []
        for size in hidden_sizes:
            lin = torch.nn.LazyLinear(size)#.to(self.device)
            relu = torch.nn.ReLU()#.to(self.device)
            norm = torch.nn.BatchNorm1d(size)#.to(self.device)
            dropout = torch.nn.Dropout(linear_dropout)#.to(self.device)
            block = torch.nn.Sequential(
                lin,relu,norm,dropout
            )
            fcs.append(block)
        self.fcs = torch.nn.ModuleList(fcs)
        self.linear = torch.nn.Linear(size,embedding_size)
        #literally this exists so the code works similar to inception feature extraction
        self.logits = torch.nn.Linear(embedding_size, 10)
        
        mname = 'basic_cnn'
        layername = lambda n,vals: '_' + n + '-'.join([str(v) for v in vals]) 
        mname += layername('fs',filter_sizes)
        mname += layername('fks',filter_kernel_sizes)
        mname += layername('h',hidden_sizes)
        mname += '_es' + '-' + str(embedding_size)
        mname += layername('drop',[initial_dropout,linear_dropout])
        self.identifier = mname
        
    def get_identifier(self):
        return self.identifier
    
    def forward(self, x):
        x = x#.to(self.device)
        x = self.init_dropout(x)
        for l in self.convs:
            x = l(x)
        x = self.linear_dropout(x)
        x = self.flatten(x)
        for l in self.fcs:
            x = l(x)
        x = self.linear(x)
        return x
 

class FacenetModel(torch.nn.Module):
    
    def __init__(self,
                 base_model = None,
                 hidden_dims = [400],
                 st_dims = [600],
                 age_dims = [400],
                 gender_dims = [400],
                 embedding_dropout=.3,
                 st_dropout = .2,
                 age_dropout = .2,
                 gender_dropout = .2,
                 base_name='model',
                 fine_tune=False,
                    ):
        super(FacenetModel,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'facenet'
        elif base_name is None:
            base_name = base_model.get_identifier()
        base_model = base_model#.to(self.device)
        for param in base_model.parameters():
            param.requires_grad = fine_tune
    
        self.base_model = base_model
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)#.to(self.device)
        curr_dim = base_model.logits.in_features
        hidden_layers = []
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)#.to(self.device)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())#.to(self.device))
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.st_layers = self.make_output(curr_dim,st_dims,10,st_dropout)
        self.age_layers = self.make_output(curr_dim,age_dims,4,age_dropout)
        self.gender_layers = self.make_output(curr_dim,gender_dims,2,gender_dropout)
        
        name_string = base_name 
        if fine_tune:
            name_string += '_finetune'
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string = add_dims(name_string,st_dims,'st')
        
        name_string = add_dims(name_string,age_dims,'a')
        name_string = add_dims(name_string,gender_dims,'g')
                    
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        name_string += '_std' + str(st_dropout).replace('0.','')
        name_string += '_ad' + str(age_dropout).replace('0.','')
        name_string += '_gd' + str(gender_dropout).replace('0.','')
                               
        self.name_string = name_string
                               
    def make_output(self,start_size,sizes,n_classes,dropout):
        layers = []
        curr_size = start_size
        for size in sizes:
            layer = torch.nn.Linear(curr_size,size)#.to(self.device)
            curr_size = size
            layers.append(layer)
            layers.append(torch.nn.ReLU())#.to(self.device))
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(curr_size,n_classes))#.to(self.device))
        layers.append(torch.nn.ReLU())#.to(self.device))
        softmax = torch.nn.Softmax(dim=-1)#.to(self.device)
        layers.append(softmax)
        return torch.nn.ModuleList(layers)
    
    def get_identifier(self):
        return self.name_string
    
    def apply_layers(self,x,layers):
        new_x = x
        for l in layers:
            new_x = l(new_x)
        return new_x
    
    def forward(self,x):
        x = x#.to(self.device)
        x = self.base_model(x)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        
        x_st = self.apply_layers(x,self.st_layers)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]