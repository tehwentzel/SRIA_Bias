import torch
import pandas as pd
import numpy as np
from scipy.ndimage import rotate
import Utils
from Utils import Constants
import cv2
from facenet_pytorch import InceptionResnetV1
import torchvision
import torchvision.models as tm

def InceptionResnet(pretrained=None,**kwargs):
    m = InceptionResnetV1(pretrained=pretrained)
    return m

def ResNet18(pretrained=True,**kwargs):
    model = tm.resnet18(pretrained=pretrained,**kwargs)
    model.logits = torch.nn.Linear(model.fc.out_features,model.fc.out_features)
    return model

def MobileNet():
    model = tm.mobilenet_v2(pretrained=True)
    dim = model.classifier[1].out_features
    model.logits = torch.nn.Linear(dim,dim)
    return model

def DenseNet():
    model = tm.densenet161(pretrained=True)
    dim = model.classifier.out_features
    model.logits = torch.nn.Linear(dim,dim)
    return model

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
 
class BaseModel(torch.nn.Module):
    
    def __init__(self):
        super(BaseModel,self).__init__()
        self.name_string = 'abstractmodel'
        
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
        layers.append(torch.nn.Linear(curr_size,n_classes))
#         layers.append(torch.nn.ReLU())
        softmax = torch.nn.Softmax(dim=-1)
        layers.append(softmax)
        return torch.nn.ModuleList(layers)
    
    def get_identifier(self):
        return self.name_string
    
    def apply_layers(self,x,layers):
        new_x = x
        for l in layers:
            new_x = l(new_x)
        return new_x
    
class FacenetModel(BaseModel):
    
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
        base_model = base_model
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
    
    
class DualFacenetModel(BaseModel):
    
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
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
        super(DualFacenetModel,self).__init__()
        
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        else:
            base_name = base_model.get_identifier()
        
        
        if feature_extractor is None:
            feature_extractor = InceptionResnetV1(pretrained='vggface2')
        for param in feature_extractor.parameters():
            param.requires_grad = fine_tune
        for param in base_model.parameters():
            param.requires_grad = True
            
        self.base_model = base_model
        self.feature_extractor = feature_extractor
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        curr_dim = base_model.logits.in_features + feature_extractor.logits.in_features
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.st_layers = self.make_output(curr_dim,st_dims,10,st_dropout)
        self.age_layers = self.make_output(curr_dim,age_dims,4,age_dropout)
        self.gender_layers = self.make_output(curr_dim,gender_dims,2,gender_dropout)
        
        name_string = 'dual_' + base_name 
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
                            
    
    def forward(self,x):
        xm = self.base_model(x)
        xf = self.feature_extractor(x)
        
        x = torch.cat((xm,xf),axis=-1)
        x = self.embedding_dropout(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x_st = self.apply_layers(x,self.st_layers)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]
    
class HistogramModel(BaseModel):
    
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
                 hidden_dims = [600],
                 st_dims = [600],
                 age_dims = [400],
                 gender_dims = [400],
                 histogram_dims = [400],
                 histogram_dropout = .01,
                 embedding_dropout=.3,
                 st_dropout = .2,
                 age_dropout = .2,
                 gender_dropout = .2,
                 base_name='model',
                 fine_tune=True,
                    ):
        super(HistogramModel,self).__init__()
        
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        elif base_name == 'model':
            try:
                base_name = base_model.get_identifier()
            except:
                base_name == 'model'
        for param in base_model.parameters():
            param.requires_grad = fine_tune

    
        self.base_model = base_model
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        
        self.histogram_dropout = torch.nn.Dropout(p=histogram_dropout)    
        histogram_layers = []
        hist_curr_dim = 300
        for i,size in enumerate(histogram_dims):
            layer = torch.nn.Linear(hist_curr_dim, size)
            hist_curr_dim = size
            histogram_layers.append(layer)
            histogram_layers.append(torch.nn.ReLU())
            
        self.histogram_layers = torch.nn.ModuleList(histogram_layers)
        
        curr_dim = base_model.logits.in_features + histogram_dims[-1]
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.st_layers = self.make_output(curr_dim,st_dims,10,st_dropout)
        self.age_layers = self.make_output(curr_dim,age_dims,4,age_dropout)
        self.gender_layers = self.make_output(curr_dim,gender_dims,2,gender_dropout)
        
        name_string = 'histogramdual_' + base_name 
        if fine_tune:
            name_string += '_finetune'
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = add_dims(name_string,histogram_dims,'hist',)
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string = add_dims(name_string,st_dims,'st')
        
        name_string = add_dims(name_string,age_dims,'a')
        name_string = add_dims(name_string,gender_dims,'g')
                    
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        name_string += '_std' + str(st_dropout).replace('0.','')
        name_string += '_ad' + str(age_dropout).replace('0.','')
        name_string += '_gd' + str(gender_dropout).replace('0.','')
                               
        self.name_string = name_string
                                 
    def forward(self,x):  
        [xx,xh] = x
        xh = self.apply_layers(xh,self.histogram_layers)
        xm = self.base_model(xx)
        xm = self.embedding_dropout(xm)
        x = torch.cat((xm,xh),axis=-1)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x_st = self.apply_layers(x,self.st_layers)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]
    
class DualHistogramModel(DualFacenetModel):
    
    def __init__(self,
                 histogram_dims = [10],
                 histogram_dropout = .1,
                 **kwargs
                    ):
        super(DualHistogramModel,self).__init__(**kwargs)
        
        self.histogram_dropout = torch.nn.Dropout(p=histogram_dropout)    
        histogram_layers = []
        hist_curr_dim = 300
        for i,size in enumerate(histogram_dims):
            layer = torch.nn.Linear(hist_curr_dim, size)
            hist_curr_dim = size
            histogram_layers.append(layer)
            histogram_layers.append(torch.nn.ReLU())
        self.histogram_layers = torch.nn.ModuleList(histogram_layers)
        
        curr_dim = self.base_model.logits.in_features + self.feature_extractor.logits.in_features + histogram_dims[-1]
        
        self.adaptor = torch.nn.Linear(curr_dim, curr_dim - histogram_dims[-1])
        self.name_string += '_hist' + '-'.join([str(d) for d in histogram_dims])
        self.name_string += '_histd_' + str(histogram_dropout)
                            
    
    def forward(self,x):
        [xx,xh] = x
        xh = self.apply_layers(xh,self.histogram_layers)
        xh = self.histogram_dropout(xh)
        
        xm = self.base_model(xx)
        xf = self.feature_extractor(xx)
        
        x = torch.cat((xm,xf),axis=-1)
        x = torch.cat((x,xh),axis=-1)
        x = self.adaptor(x)
        x = self.embedding_dropout(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x_st = self.apply_layers(x,self.st_layers)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]

    
class SimpleEncoder(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
                 hidden_dims = [500,500],
                 embedding_dropout=.3,
                 base_name='model',
                 fine_tune=True,
                 **kwargs):
        super(SimpleEncoder,self).__init__()
                               
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        for param in base_model.parameters():
            param.requires_grad = fine_tune
            
        self.base_model = base_model
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        curr_dim = base_model.logits.in_features
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        
        self.embedding_size = hidden_dims[-1]
        self.norm = torch.nn.BatchNorm1d(self.embedding_size)
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'dualencoder_' + base_name
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        self.name_string=name_string
        
    def forward(self,x):
        x = self.base_model(x)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.norm(x)
        return x
    

        
    
class TripletFacenetEncoder(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
                 hidden_dims = [400],
                 embedding_dropout=.3,
                 base_name='model',
                 fine_tune=False,
                 **kwargs):
        super(TripletFacenetEncoder,self).__init__()
                               
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        
        if feature_extractor is None:
            feature_extractor = InceptionResnetV1(pretrained='vggface2')
        for param in feature_extractor.parameters():
            param.requires_grad = fine_tune
        for param in base_model.parameters():
            param.requires_grad = True
            
        self.base_model = base_model
        self.feature_extractor = feature_extractor
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        curr_dim = base_model.logits.in_features + feature_extractor.logits.in_features
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        
        self.embedding_size = hidden_dims[-1]
        self.norm = torch.nn.BatchNorm1d(self.embedding_size)
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'dualencoder_' + base_name
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        self.name_string=name_string
        
    def forward(self,x):
        xb = self.base_model(x)
        xf = self.feature_extractor(x)
        x = torch.cat((xb,xf),axis=-1)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
class FrozenDualEncoder(TripletFacenetEncoder):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,base_model = None,**kwargs):
        if base_model is None:
            base_model = ResNet18()
        super(FrozenDualEncoder,self).__init__(base_model=base_model,**kwargs)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.name_string= 'frozen_' + self.name_string
        
class TripletFacenetClassifier(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 input_dim,
                 st_dims = [600],
                 age_dims = [400],
                 gender_dims = [400],
                 st_dropout = .2,
                 age_dropout = .2,
                 gender_dropout = .2,
                 **kwargs):
        super(TripletFacenetClassifier,self).__init__()
                               
        self.st_layers = self.make_output(input_dim,st_dims,10,st_dropout)
        self.age_layers = self.make_output(input_dim,age_dims,4,age_dropout)
        self.gender_layers = self.make_output(input_dim,gender_dims,2,gender_dropout)
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'triplet_decoder_'
        name_string = add_dims(name_string,st_dims,'st')
        
        name_string = add_dims(name_string,age_dims,'a')
        name_string = add_dims(name_string,gender_dims,'g')
        name_string += '_std' + str(st_dropout).replace('0.','')
        name_string += '_ad' + str(age_dropout).replace('0.','')
        name_string += '_gd' + str(gender_dropout).replace('0.','')
        self.name_string = name_string
        
    def embed(self,x):
        x = self.base_model(x)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return x
        
    def forward(self,x):
        x_st = self.apply_layers(x,self.st_layers)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]
    
class TripletClassifierRegressor(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 input_dim,
                 st_dims = [400,100],
                 age_dims = [400],
                 gender_dims = [400],
                 st_dropout = .2,
                 age_dropout = .2,
                 gender_dropout = .2,
                 **kwargs):
        super(TripletClassifierRegressor,self).__init__()
                               
        self.st_layers = self.make_output(input_dim,st_dims,1,st_dropout,use_softmax=False)
        self.age_layers = self.make_output(input_dim,age_dims,4,age_dropout)
        self.gender_layers = self.make_output(input_dim,gender_dims,2,gender_dropout)
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'decodewithrregression_'
        name_string = add_dims(name_string,st_dims,'st')
        
        name_string = add_dims(name_string,age_dims,'a')
        name_string = add_dims(name_string,gender_dims,'g')
        name_string += '_std' + str(st_dropout).replace('0.','')
        name_string += '_ad' + str(age_dropout).replace('0.','')
        name_string += '_gd' + str(gender_dropout).replace('0.','')
        self.name_string = name_string
        
    def embed(self,x):
        x = self.base_model(x)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return x
        
    def forward(self,x):
        x_st = self.apply_layers(x,self.st_layers)
        x_st = torch.clamp(x_st,min=0,max=9)
        x_age = self.apply_layers(x,self.age_layers)
        x_gender = self.apply_layers(x,self.gender_layers)
        return [x_st,x_age,x_gender]
    
    def make_output(self,start_size,sizes,n_classes,dropout,use_softmax=True):
        layers = []
        curr_size = start_size
        for size in sizes:
            layer = torch.nn.Linear(curr_size,size)#.to(self.device)
            curr_size = size
            layers.append(layer)
            layers.append(torch.nn.ReLU())#.to(self.device))
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(curr_size,n_classes))
        if use_softmax:
            softmax = torch.nn.Softmax(dim=-1)
            layers.append(softmax)
        return torch.nn.ModuleList(layers)
    
    
class TripletModel(BaseModel):
    
    def __init__(self,encoder=None,decoder=None):
        super(TripletModel,self).__init__()
        if encoder is None:
            encoder = TripletFacenetEncoder()
        if decoder is None:
            decoder = TripletFacenetClassifier(encoder.embedding_size)
        self.encoder = encoder
        self.decoder = decoder
        self.name_string = encoder.get_identifier() + decoder.get_identifier()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class TripletModel2(BaseModel):
    
    def __init__(self,encoder=None,decoder=None,name=''):
        super(TripletModel2,self).__init__()
        if encoder is None:
            encoder = TripletFacenetEncoder()
        if decoder is None:
            decoder = TripletFacenetClassifier(encoder.embedding_size)
        self.encoder = encoder
        self.decoder = decoder
        self.name_string = name + encoder.get_identifier() + decoder.get_identifier()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class TripletHistogramEncoder(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
                 hidden_dims = [600],
                 hist_dim=100,
                 embedding_dropout=.5,
                 base_name='histmodel',
                 fine_tune=False,
                 **kwargs):
        super(TripletHistogramEncoder,self).__init__()
                               
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        else:
            base_name = base_model.get_identifier()
        
        
        if feature_extractor is None:
            feature_extractor = InceptionResnetV1(pretrained='vggface2')
        for param in feature_extractor.parameters():
            param.requires_grad = fine_tune
        for param in base_model.parameters():
            param.requires_grad = True
            
        self.base_model = base_model
        self.feature_extractor = feature_extractor
        
        self.hist_layer =torch.nn.Linear(300,hist_dim)
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        curr_dim = base_model.logits.in_features + feature_extractor.logits.in_features + hist_dim
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        
        self.embedding_size = hidden_dims[-1]
        self.norm = torch.nn.BatchNorm1d(self.embedding_size)
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'dualencoder_' + base_name
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        self.name_string=name_string
        
    def forward(self,x):
        with torch.no_grad():
            x.requires_grad_(False)
            x, xh = add_batch_histogram(torch.clone(x),device = x.get_device())
        x.requires_grad_(True)
        xh.requires_grad_(False)
                                        
        xb = self.base_model(x)
        xf = self.feature_extractor(x)
        xh = self.hist_layer(xh)

        x = torch.cat((xb,xf,xh),axis=-1)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
class TripletHistogramSimpleEncoder(BaseModel):
    #model to use as a triplet loss
    #will tak in list of three image batchs
    #returns list of tree embeedidng batchs + predictions on first batch of images
    def __init__(self,
                 base_model = None,
                 feature_extractor = None,
                 hidden_dims = [500],
                 hist_dim=20,
                 embedding_dropout=.4,
                 base_name='histmodel',
                 fine_tune=False,
                 **kwargs):
        super(TripletHistogramSimpleEncoder,self).__init__()
                               
        if base_model is None:
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_name = 'dualfacenet'
        else:
            base_name = base_model.get_identifier()
        
        for param in base_model.parameters():
            param.requires_grad = True
            
        self.base_model = base_model
        
        self.hist_layer =torch.nn.Linear(300,hist_dim)
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        curr_dim = base_model.logits.in_features + hist_dim
        hidden_layers = []
        
        for i,size in enumerate(hidden_dims):
            layer = torch.nn.Linear(curr_dim, size)
            curr_dim = size
            hidden_layers.append(layer)
            hidden_layers.append(torch.nn.ReLU())
            
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        
        self.embedding_size = hidden_dims[-1]
        self.norm = torch.nn.BatchNorm1d(self.embedding_size)
        
        def add_dims(n,dims,prefix):
            for dim in dims:
                n += '_'+prefix+str(dim)
            return n
        
        name_string = 'simplehistencoder_' + base_name
        name_string = add_dims(name_string,hidden_dims,'h')
        name_string += '_ed' + str(embedding_dropout).replace('0.','')
        self.name_string=name_string
        
    def forward(self,x):
        with torch.no_grad():
            x.requires_grad_(False)
            x, xh = add_batch_histogram(torch.clone(x),device = x.get_device())
        x.requires_grad_(True)
        xh.requires_grad_(False)
                                        
        xb = self.base_model(x)
        xh = self.hist_layer(xh)

        x = torch.cat((xb,xh),axis=-1)
        x = self.embedding_dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.norm(x)
        return x
    

    
def add_batch_histogram(xbatch,device=None,grad=None):
    xh = torch_color_histogram(torch.clone(xbatch))
    xxb = torch.clone(xbatch)
    if device is not None:
        xh = xh.to(device)
        xxb = xxb.to(device)
    if grad is not None:
        xh.requires_grad_(grad)
        xxb.requires_grad_(grad)
    return [xxb,xh]

def apply_along_axis(function, x, axis: int = 0):
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)

def torch_histogram(image,channels=3):
    histograms = []
    for channel in range(channels):
        i = image[channel]
        i = i/torch.linalg.norm(i,ord=2)
        hist = torch.histc(i)
        histograms.append(hist)
    return torch.cat(histograms,-1)

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def torch_color_histogram(images):
    if images.ndim <= 3:
        return torch_histogram(images)
    hist = apply_along_axis(torch_histogram,images,0)
    
    return hist