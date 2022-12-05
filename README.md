# Details

This code repo contains the basic code using in the analysis.

 * Utils.py contains some contants such as the names and order of the labels used, as well as the save paths.  results_folder and data_folder are locations where the data is saved (which is different than the main repo as this code was run on a server), and thus should be changed if run by the user. It also contains several helper functions
 
 * Models.py contains the code for the different generic model types that I experimented with. Most final models used "TripletModel2"
 
 * DataLoaders.py contains the code for the files to are used to load and augment the images during training
 
 * DataFormatting contains the basic code for loading the test data into a csv file used later
 
 * TripletModels.ipynb contains the notebook that allows for training with models.  If embedding weight is set to 0, no triplet loss is used.
 
 * TripletModelsV2.ipynb contains the notebook that uses the alternative triplet loss (with 3 anchors and 3 biases, 1 for each task)
 
 * ResultsAnalysis.ipynb contains codes used to generate paper figures and evaluate the models.
 
 * <test/valdation/train>_data_clean.csv contains the images with labels one-hot encoded and results from face identification
 * <test/validation/train>_data_augmented_balanceddual.csv contains the data with anchor and bias weights pre-calculated. balanceddual vs balanceddualhistogram refers to the earlier model used.  -histogram is an older version that also passed in image color histograms into the data in an attempt to improve skin tone performance.
 

 ### depricated/old files
 
 * Unsupervised.ipynb contains earlier code used to train the models using triplet loss on only the origina images with data augmentation. 
 * BaseModels.ipynb contains code used during the baseline analysis before I coded the triplet loss. This is depricated as TripletModels contains cases for skipping the triplet loss training. This also contains some earlier code to look at the subgroup distribution.
 * Resnet.py contains code for a resnet model when I started writing the code in keras
 * models contains per-trained keras models that are not used
 
 ### Required packages

 * Pytorch 
 * pandas 
 * numpy
 * matplotlib
 * seaborn 
 * CV2
 * facenet-pytorch (pip install facenet pytorch)
 
 Facenet model is used for the pretrained models.  the original code repo is here:
 https://github.com/timesler/facenet-pytorch