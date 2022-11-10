import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product

class Constants():
    
    data_root = '../../data/'
    model_folder = data_root + 'models/'
    result_folder = data_root + 'results/'
    labels = ['skin_tone','age','gender']
    n_classes = {
        'skin_tone': 10,
        'age': 4,
        'gender': 2
    }
    default_size = 256
    resnet_size = 160
    
class ConfusionMatrixDisplay:

    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(
        self,
        *,
        include_values=True,
        cmap="Blues",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        cm_unnormalized=None,
        colorbar=True,
        title=None,
        im_kw=None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        if cm_unnormalized is None:
            cm_unnormalized = cm
            
#         maxval = cm.max()
#         cm_rgb = np.stack([maxval*np.ones(cm.shape),maxval*np.ones(cm.shape),maxval - cm],axis=2)
#         print(cm_rgb.shape)
#         for i in range(cm.shape[0]):
#             cm_rgb[i,i,0] = maxval -cm[i,i]
#             cm_rgb[i,i,1] = maxval - cm[i,i]
#             cm_rgb[i,i,2] = maxval
        
        n_classes = cm.shape[0]

        default_im_kw = dict(interpolation="nearest", cmap=cmap)
        im_kw = im_kw or {}
        im_kw = {**default_im_kw, **im_kw}

        self.im_ = ax.imshow(cm, **im_kw)
#         self.im_ = ax.imshow(cm_rgb, **im_kw)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(1.0)

        
        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm_unnormalized[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm_unnormalized[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm_unnormalized[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
        if title is not None:
            ax.set_title(title)
        self.figure_ = fig
        self.ax_ = ax
        return self

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="Blues",
        ax=None,
        colorbar=True,
        title=None,
        im_kw=None,
    ):
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred,axis=1)
            
        if display_labels is None:
            if labels is None:
                display_labels = list(set(np.unique(y_true)).union(set(np.unique(y_pred))) )
            else:
                display_labels = labels

        cm = confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            normalize=normalize,
        )
        
        cm_unnormalized = confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
        )

        disp = cls(confusion_matrix=cm, display_labels=display_labels)

        return disp.plot(
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            cm_unnormalized=cm_unnormalized,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=colorbar,
            title=title,
            im_kw=im_kw,
        )
    
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