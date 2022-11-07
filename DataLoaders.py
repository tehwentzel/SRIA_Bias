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

def upsample_data(df,fit_df=None,**kwargs):
    if fit_df is None:
        fit_df = df.copy()
        
    fit_df = add_group(fit_df,**kwargs)
    df = add_group(df,**kwargs)

    ratios = calc_label_upsample(fit_df)
    df['group_ratio'] = df.group.apply(lambda x: ratios.get(x,1))
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

class Augmentor():
    
    def __init__(self,image_size = None,noise_sigma=.03):
        if image_size is None:
            image_size = Constants.resnet_size
        self.image_size = image_size
        self.noise_sigma = noise_sigma
        
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
        angle = (self.random_range(.01,.99)*360 - 180)
        rgb = len(img.shape) == 3
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
        img = rotate(img, angle,reshape=False)
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
        
class FaceGeneratorIterator(torch.utils.data.Dataset):
    
    def __init__(self,df,root,
                 filter_nonfaces=True,
                 batch_size=1,
                 image_shape=None,
                 labels=None,
                 regularize_labels=False,
                 upsample=True,
                 fit_df=None,
                 shuffle_on_init=True,
                 validation = False,
                 preload=False,
                 **kwargs):
        super(FaceGeneratorIterator,self).__init__()
        df = df.copy()
        if fit_df is None:
            fit_df = df.copy()
        if filter_nonfaces:
            df = df[df.is_face]
            fit_df = fit_df[fit_df.is_face]
        if upsample:
            df = upsample_data(df,fit_df=fit_df)
        if shuffle_on_init:
            df = df.sample(frac=1)
        self.df = df
        
        self.image_shape = image_shape if image_shape is not None else Constants.resnet_size
        self.batch_size = 1
        self.root=root
        
        self.labels = labels if (labels is not None) else Constants.labels
        
        self.n_classes = {label: len(self.df[label].unique()) for label in self.labels}
        self.augmentor = Augmentor(**kwargs)
        if regularize_labels:
            self.df[self.labels] = self.df[self.labels].values/self.df[self.labels].max().values
        
        
        self.shuffle_on_epoch = not validation
        self.augment_images = not validation
        
        if preload:
            self.df['image'] = self.df.name.apply(self.process_image_file)
        self.preloaded = preload
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

def FaceGenerator(labels,data_root,batch_size=100, **kwargs):
    #Is this legal?
    dataset = FaceGeneratorIterator(labels,data_root,**kwargs)
    print(dataset.df.shape)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=dataset.shuffle_on_epoch)