import numpy as np
import matplotlib.pyplot as plt

class Constants():
    
    data_root = '../../data/'
    model_folder = data_root + 'models/'
    result_folder = data_root + 'resuls/'
    labels = ['skin_tone','age','gender']
    n_classes = {
        'skin_tone': 10,
        'age': 4,
        'gender': 2
    }
    default_size = 256
    resnet_size = 160
    
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def resnet_preprocess(x): 
    #https://github.com/kentsommer/keras-inceptionV4/issues/5#issuecomment-287673694
    x = 2*(x-1) 
    return x

def plot_selection(images,rows=20,columns=3):
    fig = plt.figure(figsize=(3*columns, 3*rows))
    # setting values to rows and column variables
    for i in range(rows*columns):
        if i >= len(images):
            return
        image = images[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(image)
        
    return