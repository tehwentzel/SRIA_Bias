import Utils
from Utils import Constants
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate

def add_group(df,labels=None,filter_face = True):
    if labels is None:
        labels = Constants.labels
    if filter_face:
        df = df[df.is_face]
    to_string = lambda row: '-'.join([str(x) for x in row])
    df = df.copy()
    df['group'] = df.apply(lambda row: to_string(row[labels]),axis=1)
    return df
    
def calc_label_upsample(df,**kwargs):
    if 'group' not in df.columns:
        df = add_group(df,**kwargs)
    counts = {}
    max_group = 0
    min_group = df.shape[0]
    for group,subdf in df.groupby('group'):
        size = subdf.shape[0]
        counts[group] = size
        max_group = max(max_group,size)
        min_group = min(min_group,size)
    #get the number of times you'd need to upsample 
    ratios = {k: 1/(v/max_group) for k,v in counts.items()} 
    return ratios

def get_upsample_weights(df,fit_df=None,**kwargs):
    if fit_df is None:
        fit_df = df.copy()
        
    fit_df = add_group(fit_df,**kwargs)
    df = add_group(df.copy(),**kwargs)

    ratios = calc_label_upsample(fit_df)
    df['group_ratio'] = df.group.apply(lambda x: ratios.get(x,1))
    return df
#     new_df = []

def upsample_data(df,fit_df=None,**kwargs):
    df = get_upsample_weights(df,fit_df=fit_df)
    new_df = []
    for i,row in df.iterrows():
        repeats = np.ceil(row.group_ratio).astype(int)
        for ii in range(repeats):
            new_df.append(row)
    return pd.DataFrame(new_df).reset_index().drop(['index','group','group_ratio'],axis=1)
    
def read_image(file,size=None):
    img = Image.open(Constants.data_root+file).convert('RGB')
    if size is not None:
        img = cv2.resize(img,(size,size))
    img = np.array(img).astype(np.float32)/255
    return img
    
def imgs_to_np(array):
    array = array.cpu().numpy()
    if array.ndim > 3:
        return np.moveaxis(array,1,-1)
    return np.moveaxis(array,0,-1)

def imgs_to_torch(array,convert=False):
    if array.ndim > 3:
        array = np.moveaxis(array,-1,1)
    else:
        array = np.moveaxis(array,-1,0)
    if convert:
        array = torch.from_numpy(array)
    return array

def array_softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def get_random_upsampler(df,fit_df=None,drop_last=True,replacement=True,softmax=False,**kwargs):
    df = get_upsample_weights(df,fit_df=fit_df)
    if softmax:
        df['group_ratio'] = df.group_ratio.apply(array_softmax)
    sampler = torch.utils.data.WeightedRandomSampler(df.group_ratio.values,df.shape[0],replacement=replacement)
    return sampler

def make_skintone_patch(level,size=None,rescale=True):
    if size is None:
        size = 160
    if level == 0:
        rgb = (246, 237, 228)
    elif level == 1:
        rgb = (243, 231, 219)
    elif level == 2:
        rgb = (247, 234, 208)
    elif level == 3:
        rgb = (234, 218, 186)
    elif level == 4:
        rgb = (215, 189, 150)
    elif level == 5:
        rgb = (160, 126, 86)
    elif level == 6:
        rgb = (130, 92,67)
    elif level == 7:
        rgb = (96, 65, 52)
    elif level == 8:
        rgb = (58,49,42)
    else:
        rgb = (41,36,32)
    img = np.stack([np.full((size,size),val) for val in rgb],axis=2)
    if rescale:
        img = img.astype(np.float32)/255
    return img

class Augmentor():
    
    def __init__(self,image_size = None,noise_sigma=.05,augment_prob = .8,**kwargs):
        if image_size is None:
            image_size = Constants.resnet_size
        self.image_size = image_size
        self.noise_sigma = noise_sigma
        self.augment_prob = augment_prob
        
    def random_range(self,min_ratio = 0.01, max_ratio = .9):
        return max(min(max_ratio,1.5*np.random.random()), min_ratio)
    
    def random_crop(self,img):
        ratio = self.random_range(min_ratio=0.5)
        crop_size = [int(img.shape[0]*ratio), int(img.shape[1]*ratio)]
        assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
        img = img.copy()
        w, h = img.shape[:2]
        x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
        img = img[y:y+crop_size[0], x:x+crop_size[1]]
        return img
    
    def random_rotation(self,img, bg_patch=(5,5)):
        assert len(img.shape) <= 3, "Incorrect image shape"
        key = np.random.random()
        #30% change of flipping along each axis before rotating
        if key < .33:
            img = np.flip(img,1)
        elif key > .66:
            img = np.flip(img,2)
#         90% change of rotating
#         if key < .9:
        angle = (self.random_range(.01,.99)*360 - 180)
        rgb = len(img.shape) == 3
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
        img = rotate(img,angle=angle)
        mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
        img[mask] = bg_color
        return img
    
    def gaussian_noise(self,img, mean=0):
        img = img.copy()
        noise = (np.random.normal(mean, self.noise_sigma, img.shape)).astype(np.float16)
        mask_overflow_upper = img+noise >= 1
        mask_overflow_lower = img+noise < 0
        noise[mask_overflow_upper] = 1
        noise[mask_overflow_lower] = 0

        img += noise
        return img
    
    
    def color_shift(self,img):
        img = img.copy().astype(np.float16)
        for channel in [0,1,2]:
            img[channel] *= .5 + self.random_range(.1,.9)
        return img
    
    def augment_image(self,img,crop=True,rotate=True,noise=True,color_shift=False):
        if np.random.random() < self.augment_prob:
            shape = img.shape
            if img.ndim < 3 or shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
                print('bad shape',shape)
            if crop:
                img = self.random_crop(img)
            if rotate:
                img = self.random_rotation(img)
            if noise:
                img = self.gaussian_noise(img)
            if color_shift:
                img = self.color_shift(img)
        img = self.format_image(img)
        return img
    
    def augment_images(self,images,**kwargs):
        images = np.stack([self.augment_image(i,**kwargs) for i in images])
        return images
    
    def format_image(self,images,normalize=False,whiten=False,resnet_normalize=False):
        images = images.astype(np.float32)
        if images.max() > 1:
            images = images/images.max()
        
        if whiten:
            images = Utils.prewhiten(images)
        if normalize:
            images = Utils.l2_normalize(images)
        if resnet_normalize:
            images = resnet_preprocess(images)
        images = cv2.resize(images,[self.image_size, self.image_size])
        return images
    
class GeneratorBase(torch.utils.data.Dataset):
    
    def __init__(self,df,root,fit_df=None,filter_nonfaces=True,shuffle_on_init=True,image_shape=None,validation=False,**kwargs):
        super(GeneratorBase,self).__init__()
        df = df.copy()
        if fit_df is None:
            fit_df = df.copy()
        if filter_nonfaces:
            df = df[df.is_face]
            fit_df = fit_df[fit_df.is_face]
#         if upsample:
#             df = upsample_data(df,fit_df=fit_df)
        if shuffle_on_init:
            df = df.sample(frac=1)
        self.fit_df = fit_df
        self.df = df
        
        self.image_shape = image_shape if image_shape is not None else Constants.resnet_size
        self.batch_size = 1
        self.root=root
        self.validation=validation
        self.shuffle_on_epoch = not validation
        self.augment_images = not validation
        
        
        self.augmentor = Augmentor(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return self.df.shape[0]
            
    
    def __getitem__(self,idx):
        subdf = self.df.iloc[idx]
        return self.process_files(subdf)
        
    def batch_subset(self,idx):
        start = idx*self.batch_size
        stop = start + self.batch_size
        subset = self.df.iloc[start:stop,:]
        return subset
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)
    
    def process_image_file(self,i):
        image = read_image(i)
        return image
    
class FaceGeneratorIterator(GeneratorBase):
    
    def __init__(self,df,root,
                 image_shape=None,
                 labels=None,
                 regularize_labels=False,
                 upsample=True,
                 preload=False,
                 **kwargs):
        super(FaceGeneratorIterator,self).__init__(df,root,**kwargs)
        if upsample:
            self.df = upsample_data(self.df,fit_df=self.fit_df)

        self.labels = labels if (labels is not None) else Constants.labels
        
        self.n_classes = {label: len(self.df[label].unique()) for label in self.labels}
        if regularize_labels:
            self.df[self.labels] = self.df[self.labels].values/self.df[self.labels].max().values
        
        if preload:
            self.df['image'] = self.df.name.apply(self.process_image_file)
        self.preloaded = preload
        
    def get_dataset(self):
        return self.process_files(self.df['name'])
    
    def process_files(self,subdf,augment_images=None,**kwargs):
        #will return a list of arrays [images, label1, label2, label3, etc]
        if self.preloaded:
            image = subdf['image']
        else:
            image = self.process_image_file(subdf['name'])
        augment_images = self.augment_images if augment_images is None else augment_images
        if augment_images:
            image = self.augmentor.augment_image(image,**kwargs)
        else:
            image = self.augmentor.format_image(image,**kwargs)
        #swaps axis to be batch x chanells x widht x height
        image = imgs_to_torch(image,convert=True)
        labels = [subdf[label] for label in self.labels]
        output = [image, labels]
        return output

def FaceGenerator(labels,data_root,
                  batch_size=100,
                  workers=2,
                  random_upsample=False,
                  fit_df=None,softmax=False, 
                  upsample=False,
                  **kwargs):
    if random_upsample:
        upsampler = get_random_upsampler(labels,fit_df=fit_df,softmax=softmax)
        upsample = False
        shuffle=False
    else:
        upsampler=None
        
    dataset = FaceGeneratorIterator(labels,data_root,fit_df=fit_df,upsample=upsample,**kwargs)
    if not random_upsample:
        shuffle = dataset.shuffle_on_epoch
    print(dataset.df.shape)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=workers)

class TripletFaceGeneratorIterator(FaceGeneratorIterator):
    
    #alternative generator that returns [image,anchor,bias, [skintone, age, gender]]
    #where anchor is high-confidance in same class, bias is a counterfactual
    #assumes I've preprocessed the  input dataframe to have anchor and bias weights for each subgroup in order skintone-age-gender
    #i havent tested if preloading works since i dont use it
    
    def __init__(self,df,root,nonface_bias_prob=.01,skintone_patch_anchor_prob=0,**kwargs):
        super(TripletFaceGeneratorIterator,self).__init__(df,root,**kwargs)
        self.nonface_df = df[~df.is_face]
        if self.nonface_df.shape[0] < 5 or nonface_bias_prob <= .0001:
            print('no faces')
            self.use_nonface = lambda : False
        else:
            self.use_nonface = lambda : np.random.random() < nonface_bias_prob
        def get_subgroup(row):
            return str(row['skin_tone']) + '-' + str(row['age']) + '-' + str(row['gender'])
        self.df['subgroup'] = self.df.apply(get_subgroup,axis=1)
        self.skintone_patch_anchor_prob = skintone_patch_anchor_prob
        
    def get_skintone_image(self,skintone):
        image = make_skintone_patch(skintone,size=self.image_shape)
        if self.augment_images:
            image = self.augmentor.gaussian_noise(image)
        image = self.augmentor.format_image(image)
        image = imgs_to_torch(image,convert=True)
        return image
        
    def process_single_image(self,subdf,augment_images=None,**kwargs):
        imagename= subdf['name']
        if self.preloaded:
            image = subdf['image']
        else:
            image = self.process_image_file(imagename)
        augment_images = self.augment_images if augment_images is None else augment_images
        if augment_images:
            image = self.augmentor.augment_image(image,**kwargs)
        else:
            image = self.augmentor.format_image(image,**kwargs)
        #swaps axis to be batch x chanells x widht x height
        image = imgs_to_torch(image,convert=True)
        return image
    
    def process_files(self,subdf,**kwargs):
        #will return a list of arrays [images, label1, label2, label3, etc]
        baseimage = self.process_single_image(subdf)
        labels = [subdf[label] for label in self.labels]
        subgroup = subdf['subgroup']
        if np.random.random() > self.skintone_patch_anchor_prob:
            anchor = self.df.sample(n=1,weights=subgroup+'_anchor').iloc[0]
            anchorimage = self.process_single_image(anchor)
        else:
            anchorimage = self.get_skintone_image(subdf['skin_tone'])
#         assert(subgroup == anchor['subgroup'])
        if self.use_nonface():
            bias = self.nonface_df.sample(n=1).iloc[0]
        else:
            bias = self.df.sample(n=1,weights=subgroup+'_bias').iloc[0]
#         assert(subgroup != bias['subgroup'])
        
        biasimage = self.process_single_image(bias)
        
        
        output = [[baseimage,anchorimage,biasimage], labels]
        return output

def TripletFaceGenerator(labels,data_root,
                         batch_size=100,
                         workers=2,
                         random_upsample=False,
                         fit_df=None, 
                         softmax=False,
                         upsample=False,**kwargs):
    if random_upsample:
        upsampler = get_random_upsampler(labels,fit_df=fit_df,softmax=softmax)
        upsample = False
        shuffle=False
    else:
        upsampler=None
    dataset = TripletFaceGeneratorIterator(labels,data_root,fit_df=fit_df,upsample=upsample,**kwargs)
    if not random_upsample:
        shuffle=dataset.shuffle_on_epoch
    print(dataset.df.shape)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=workers)

class TripletFaceGeneratorIterator2(TripletFaceGeneratorIterator):
    
    #alternative generator that returns [image,anchor,bias, [skintone, age, gender]]
    #where anchor is high-confidance in same class, bias is a counterfactual
    #assumes I've preprocessed the  input dataframe to have anchor and bias weights for each subgroup in order skintone-age-gender
    #i havent tested if preloading works since i dont use it
    
    def __init__(self,df,root,nonface_bias_prob=.01,skintone_patch_anchor_prob=0,**kwargs):
        super(TripletFaceGeneratorIterator2,self).__init__(df,root,**kwargs)
        
    
    def process_files(self,subdf,**kwargs):
        #will return a list of arrays [images, label1, label2, label3, etc]
        baseimage = self.process_single_image(subdf)
        subgroup = subdf['subgroup']
        base_weights  = self.df[subgroup + '_anchor'] + self.df[subgroup + '_bias']
        labels = []
        anchors = []
        biases = []
        for label in self.labels:
            y = subdf[label]
            in_class = self.df[label].apply(lambda x: x == y)
            not_in_class = self.df[label].apply(lambda x: x != y)
            anchor_weights = base_weights * in_class
            bias_weights = base_weights * not_in_class
            
            if np.random.random() > self.skintone_patch_anchor_prob:
                anchor = self.df.sample(n=1,weights=anchor_weights).iloc[0]
                anchorimage = self.process_single_image(anchor)
            else:
                anchorimage = self.get_skintone_image(subdf['skin_tone'])
    #         assert(subgroup == anchor['subgroup'])
            if self.use_nonface():
                bias = self.nonface_df.sample(n=1).iloc[0]
            else:
                bias = self.df.sample(n=1,weights=bias_weights).iloc[0]

            biasimage = self.process_single_image(bias)
            labels.append(y)
            anchors.append(anchorimage)
            biases.append(biasimage)
        output = [baseimage, anchors, biases, labels]
        return output

def TripletFaceGenerator2(labels,data_root,
                         batch_size=100,
                         workers=2,
                         random_upsample=False,
                         fit_df=None, 
                         softmax=False,
                         upsample=False,**kwargs):
    if random_upsample:
        upsampler = get_random_upsampler(labels,fit_df=fit_df,softmax=softmax)
        upsample = False
        shuffle=False
    else:
        upsampler=None
    dataset = TripletFaceGeneratorIterator2(labels,data_root,fit_df=fit_df,upsample=upsample,**kwargs)
    if not random_upsample:
        shuffle=dataset.shuffle_on_epoch
    print(dataset.df.shape)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=workers)

class UnsupervisedTripletGeneratorIterator(FaceGeneratorIterator):
    
    #alternative generator that returns [image,anchor,bias, [skintone, age, gender]]
    #where anchor is high-confidance in same class, bias is a counterfactual
    #assumes I've preprocessed the  input dataframe to have anchor and bias weights for each subgroup in order skintone-age-gender
    #i havent tested if preloading works since i dont use it
    
    def __init__(self,df,
                 root,**kwargs):
        if isinstance(df,list):
            df = pd.concat(df,axis=0,ignore_index=True)
        super(UnsupervisedTripletGeneratorIterator,self).__init__(df,root,augment_prob=1,**kwargs)
    
    def process_single_image(self,subdf,**kwargs):
        imagename= subdf['name']
        image = self.process_image_file(imagename)
        if self.validation:
            images = self.augmentor.format_image(image,**kwargs)
        else:
            image = self.augmentor.augment_image(image,**kwargs)
        #swaps axis to be batch x chanells x widht x height
        return image
   
    def process_files(self,subdf,**kwargs):
        #will return a list of arrays [image,image,otherimage]
        #image instances should have different augmentation
        baseimage = self.process_single_image(subdf)
        labels = [subdf[label] for label in self.labels]
        if self.validation:
            return imgs_to_torch(baseimage,convert=True), labels
        
        bias = self.df.sample(n=1).iloc[0]
        while bias['name'] == subdf['name']:
            bias = self.df.sample(n=1).iloc[0]
        anchorimage = self.process_single_image(subdf)
        biasimage = self.process_single_image(bias)
        
        images = [baseimage,anchorimage,biasimage]
        images = [imgs_to_torch(i,convert=True) for i in images]
        
        output = [images,labels]
        return output
    
def UnsupervisedTripletGenerator(labels,data_root,batch_size=100, **kwargs):
    #Is this legal?
    dataset = UnsupervisedTripletGeneratorIterator(labels,data_root,**kwargs)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=dataset.shuffle_on_epoch)
