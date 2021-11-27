#%%
# Import libraries
#######################################################################


# from logging import CSV
import os
import glob
# from posixpath import splitext
# from re import T
import time
from zipfile import ZipFile
import random
import datetime
import shutil
import gc

import numpy as np
from numpy.core.numeric import ones
# from numpy.core.numeric import moveaxis
import pandas as pd
from math import dist
from sklearn import neighbors
# from pandas.core.indexes.base import Index
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


import rasterio
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
matplotlib.rcParams.update({'font.family': 'Times New Roman'})#,'text.usetex': True})
# matplotlib.rc('xtick', labelsize=14) 
# matplotlib.rc('ytick', labelsize=14)
import seaborn as sns
# from matplotlib import figure
# from scipy import ndimage as ndi
from skimage.feature import peak_local_max
# from skimage import exposure
from skimage import img_as_ubyte

import cv2
# from torch.hub import get_dir
import albumentations as A
import torch


from deepforest import main
# from deepforest import get_data
from deepforest import preprocess
from deepforest import utilities
from deepforest import visualize
from deepforest import evaluate

from pytorch_lightning.loggers import CSVLogger


from pandas_profiling import ProfileReport
# import splitfolders 

rs=42


import warnings
warnings.filterwarnings("ignore")

################################################################################################
# FUNCTIONS
################################################################################################


#######################################################################
# Split dataset
#######################################################################

def train_val_test_split(annotations, vr, tr):

    np.random.seed(44)
    print("########### Train Test Val Script started ###########")

    val_ratio = vr
    test_ratio = tr

    src = 'Agisoft/Ortho' +"//"  # Folder to copy images from
    root_dir = src
    allFileNames_temp = glob.glob(src+'*.tif')

    allFileNames =[]
    for i in allFileNames_temp:
        allFileNames.append(os.path.basename(i))

    # allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_paths, val_paths, test_paths = np.split(np.array(allFileNames),
                                                                [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                                int(len(allFileNames) * (1 - val_ratio)),
                                                                ])

    train_FileNames = [src + '//' + name for name in train_paths.tolist()]
    val_FileNames = [src + '//' + name for name in val_paths.tolist()]
    test_FileNames = [src + '//' + name for name in test_paths.tolist()]

    print('Total images: '+ str(len(allFileNames)))
    print('Training: '+ str(len(train_FileNames)))
    print('Validation: '+  str(len(val_FileNames)))
    print('Testing: '+ str(len(test_FileNames)))

    # # Creating Train / Val / Test folders (One time use)

    tr_ortho_dir = root_dir + '/train//'
    val_ortho_dir = root_dir + '/val//'
    te_ortho_dir = root_dir + '/test//'

    if os.path.isdir(tr_ortho_dir) == False:
        os.makedirs(tr_ortho_dir)
    if os.path.isdir(val_ortho_dir) == False:
        os.makedirs(val_ortho_dir)
    if os.path.isdir(te_ortho_dir) == False:
        os.makedirs(te_ortho_dir)

    # Clean existing folders from previous runs
    tr_files = glob.glob('%s*' %tr_ortho_dir)
    for f in tr_files:
        os.remove(f)

    val_files = glob.glob('%s*' %val_ortho_dir)
    for f in val_files:
        os.remove(f)

    te_files = glob.glob('%s*' %te_ortho_dir)
    for f in te_files:
        os.remove(f)

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, tr_ortho_dir)

    for name in val_FileNames:
        shutil.copy(name, val_ortho_dir)

    for name in test_FileNames:
        shutil.copy(name, te_ortho_dir)

    # Split annotations
    # image_paths = annotations.image_path.unique()
    # train_paths = np.random.choice(image_paths, int(len(image_paths)*train_split))
    train_annotations = annotations.loc[annotations.image_path.isin(train_paths)]
    val_annotations = annotations.loc[annotations.image_path.isin(val_paths)]
    test_annotations = annotations.loc[annotations.image_path.isin(test_paths)]

    train_file= os.path.join('Agisoft/Ortho/train/',"train.csv")
    val_file= os.path.join('Agisoft/Ortho/val/',"val.csv")
    test_file= os.path.join('Agisoft/Ortho/test/',"test.csv")

    train_annotations.to_csv(train_file,index=False)
    val_annotations.to_csv(val_file,index=False)
    test_annotations.to_csv(test_file,index=False)

    print("########### Train Test Val Script Ended ###########")

    return train_annotations, val_annotations, test_annotations


#######################################################################
# Create annotations
#######################################################################

def unzip(file_name,save_path, save_name):

    """Unzip all .xml annotation files and create a single dataframe


    Returns: 
        annotations dataframe
    """

    # # Unzip annotations file
    # path = "CVAT/" + file_name
    # with ZipFile(path, 'r') as zip:
    #     zip.extractall()

    # Get all .xml annotation files
    annotations_xml=glob.glob("Annotations\*.xml")

    # Combine .xml annotation files into a single DataFrame 
    for idx, file in enumerate(annotations_xml):
        annotations_individual = utilities.xml_to_annotations(file)
        if idx == 0:
            annotations = annotations_individual
        else:
            frames=[annotations, annotations_individual]
            annotations = pd.concat(frames)


    # # Convert the entire annotations DataFrame to a .csv file for DeepForest
    annotations.to_csv(os.path.join(save_path, save_name), index=False)

    return annotations



# #######################################################################
# # Train test split of full trays
# #######################################################################


# def ortho_train_test_split (train_split, annotations):

#     image_paths = annotations.image_path.unique()

#     train_paths = np.random.choice(image_paths, int(len(image_paths)*train_split))
#     train_annotations = annotations.loc[annotations.image_path.isin(train_paths)]
#     test_annotations = annotations.loc[~annotations.image_path.isin(train_paths)]

#     train_file= os.path.join('Agisoft/Ortho/',"train.csv")
#     test_file= os.path.join('Agisoft/Ortho/',"test.csv")

#     train_annotations.to_csv(train_file,index=False)
#     test_annotations.to_csv(test_file,index=False)

#     return train_annotations, test_annotations, test_file

#######################################################################
# Create image tiles of training trays
#######################################################################

def tiles (image , size, overlap, dir):
    """Creates tiles from an image


    Returns:
        - image file as RGB numpy array
        - windows used to generate tiles

    """
   
    rasterO = "Agisoft/Ortho/" + image
    img=rasterio.open(rasterO).read()

    # Reshuffle array such that the number of channels is the last element as required by DF 
    img=np.moveaxis(img, 0, 2)

    # Compute coordinates for windows (windows will form tiles)
    windows = preprocess.compute_windows(img, size, overlap)

    # Crop image according to windows
    # get list of crops (returns a list of arrays)
    crop =[]
    for window in windows:
        crop.append(window.apply(img))

    # Save ctopped images i.e. tiles
    for idx, c in enumerate(crop):
        preprocess.save_crop(dir, image, idx, c)

    return img, windows


def tile_annotations (windowsO, annotations):

    sel_annotations=[]
    sel_annotations_df = pd.DataFrame()
    for idx, windows in enumerate (windowsO):
        for i in range(len(windows)):
            crop_annotations=preprocess.select_annotations(annotations[annotations.image_path == annotations.image_path.unique()[idx]], windows, i)
            sel_annotations.append(crop_annotations)

        for j in range(len(sel_annotations)):
            if j == 0:
                sel_annotations_df = sel_annotations[j]
            else:
                frames=[sel_annotations_df, sel_annotations[j]]
                sel_annotations_df = pd.concat(frames)

        if idx == 0:
            t_annotations = sel_annotations_df
        else:
            frames=[t_annotations, sel_annotations_df]
            t_annotations = pd.concat(frames)

    return t_annotations


def plot_predictions_from_df(df, img, colour = (255, 255, 0)):

    # Draw predictions on BGR 
    image = img[:,:,::-1]
    image=img
    predicted_raster_image = visualize.plot_predictions(image, df, color=colour)

    return predicted_raster_image

#######################################################################
# Augmentation
#######################################################################

def augment (annotations, number, dir, augs, min_vis):
    """ Augment the images in the Train/ directory
        Save augmented images to the same path

    Returns:
        Annotations dataframe included augmented annotations
    """
    random.seed(44)


    augmented_path=dir # path to store aumented images
    # images=[] # to store paths of images from folder

    image_list=glob.glob("%s*.png" %dir)
    
    if augs == 'all':
        #  Define the transform
        transform = A.Compose([
            A.ShiftScaleRotate(always_apply=False, p=0.8, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-89, 89), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            A.HorizontalFlip(p=0.8),
            A.RandomBrightnessContrast(always_apply=False, p=0.8, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=True),
            A.RandomRotate90(always_apply=False, p=0.8)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'SSR':
        #  Define the transform
        transform = A.Compose([
            A.ShiftScaleRotate(always_apply=False, p=0.8, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-89, 89), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'HF_RBC':
        #  Define the transform
        transform = A.Compose([
            A.HorizontalFlip(p=0.8),
            A.RandomBrightnessContrast(always_apply=False, p=0.8, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=True),
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'HF':
        #  Define the transform
        transform = A.Compose([
            A.HorizontalFlip(p=0.8)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'RBC':
        #  Define the transform
        transform = A.Compose([
            A.RandomBrightnessContrast(always_apply=False, p=1, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=True)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'RR':
        #  Define the transform
        transform = A.Compose([
            A.RandomRotate90(always_apply=False, p=0.8)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_vis))
    elif augs == 'none':
        print('No augmentations required')
    else:
        print('No valid augmentation selected')
    images_to_generate=number  #you can change this value according to your requirement
    i=1                        # variable to iterate till images_to_generate

    if augs != 'none':
        while i<=images_to_generate:
            image=random.choice(image_list)
            original_image = cv2.imread(image)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_annotations= annotations.loc[(annotations.image_path == os.path.basename(image)),('xmin', 'ymin','xmax','ymax','label')]

            # perform transformation
            transformed = transform(image=original_image, bboxes=original_annotations.values.tolist())#,min_visibility=0.8)

            image_name=os.path.splitext(os.path.basename(image))[0]
            new_image_path= "%s%s_augmented_%s.png" %(augmented_path,image_name, i)
            transformed_image = img_as_ubyte(transformed['image'])  #Convert an image to unsigned byte format, with values in [0, 255].
            # transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
            cv2.imwrite(new_image_path, transformed_image) # save transformed image to path

            augmented_annotations = pd.DataFrame(transformed['bboxes'], columns=('xmin', 'ymin','xmax','ymax','label'), dtype='int')
            augmented_annotations.insert(loc=0, column='image_path', value=os.path.basename(new_image_path))
            framesA=[annotations, augmented_annotations]
            annotations = pd.concat(framesA)

            i += 1  

    return annotations

# #######################################################################
# # DeepForest: TRAIN/VALIDATE SPLIT
# #######################################################################

# def train_valid_split (annotations, validation_split, Train_dir):
#     """ Splits the images into test and validation sets
#         Creates  and saves train_annotations and validation_annotations .csv files for training

#     Returns:
#         None
#     """
#     validation_split=0.25

#     #  Split "annotations" dataframe int test and validate
#     # Randomly select "validate_split" percentage of annotations for validation
#     image_paths = annotations.image_path.unique()

#     validation_paths = np.random.choice(image_paths, int(len(image_paths)*validation_split))
#     validation_annotations = annotations.loc[annotations.image_path.isin(validation_paths)]
#     train_annotations = annotations.loc[~annotations.image_path.isin(validation_paths)]

#     print("There are {} training seedling annotations".format(train_annotations.shape[0]))
#     print("There are {} test seedling annotations".format(validation_annotations.shape[0]))

#     #save to file and create the file dir
#     annotations_file= os.path.join(Train_dir,"train.csv")
#     validation_file= os.path.join(Train_dir,"validation.csv")
#     #Write window annotations file without a header row, same location as the "base_dir" above.
#     train_annotations.to_csv(annotations_file,index=False)
#     validation_annotations.to_csv(validation_file,index=False)

#     return annotations_file, validation_file


def save_annotations (annotations, dir):
    if dir == 'Train/':
        file= os.path.join(dir,"train.csv")
    if dir == 'Val/':
        file= os.path.join(dir,"validation.csv")
    annotations.to_csv(file,index=False)
    return file

#######################################################################
# Deep Forest - CONFIGURE
#######################################################################

def config (annotations_file, validation_file, batch_size, nms_threshold, score_threshold, train_epochs, train_learning_rate, validation_iou_threshold,optimiser,learn_rate,file_name,model_path=None):

    # initialise the model and change the corresponding config file
    m = main.deepforest()
    #load the lastest release model 
    m.use_release()
    m.to("cuda")
    m.config["workers"] = 0
    m.config['gpus'] = 1 #move to GPU and use all the GPU resources
    m.config["batch_size"] = batch_size

    m.config["nms_thresh"] = nms_threshold
    m.config["score_thresh"] = score_threshold

    m.config["train"]["csv_file"] = annotations_file
    m.config["train"]["root_dir"] = os.path.dirname(annotations_file)

    m.config["train"]['epochs'] = train_epochs
    m.config["train"]['lr'] = train_learning_rate

    m.config["train"]["optimiser"] = optimiser
    m.config["train"]["lr_schedule"] = learn_rate

    m.config["validation"]["csv_file"] = validation_file
    m.config["validation"]["root_dir"] = os.path.dirname(validation_file)
    m.config["validation"]["iou_threshold"] = validation_iou_threshold

    # m.config["train"]["fast_dev_run"] = True
    # file_name = optimiser + '_' + learn_rate
    logger=CSVLogger('logs',name=file_name)
    
    #create a pytorch lighting trainer used to training 
    m.create_trainer(logger=logger)
    # m.create_trainer(logger=True)

    return m



#######################################################################
# DeepForest: TRAIN and evaluate
#######################################################################

def train (m):
    """Trains DeepForest model
        Times the training

    Returns:
        Trained DeepForest model
    """

    start_time = time.time()
    m.trainer.fit(m)
    print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")

    return m

#######################################################################
# Deep Forest: EVALUATE TRAINING
#######################################################################
def DFeval (m, annotations_file, thresh,save_dir):

    save_dir = os.path.join(os.getcwd(),'Results')
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    results = m.evaluate(annotations_file, os.path.dirname(annotations_file), savedir= save_dir, iou_threshold = thresh)

    return results


def clean_training():
    # Empty training folder between iterations
    train_files = glob.glob('Train/*.png')
    for f in train_files:
        os.remove(f)


def clean_test():
    # Empty training folder between iterations
    test_files = glob.glob('Test/*.tif')
    for f in test_files:
        os.remove(f)


def clean_annotations ():
    # Empty training folder between iterations
    annotations_files = glob.glob('Annotations/*')
    for f in annotations_files:
        os.remove(f)


def clean_tiles ():
    # training data
    train_all = glob.glob('Train/*.png')
    train_augmented =  glob.glob('Train/*augmented*.png')
    for element in train_augmented:
        if element in train_all:
            train_all.remove(element)
    tiles = train_all
    for f in tiles:
        os.remove(f)

    # val data
    val_all = glob.glob('Val/*.png')
    val_augmented =  glob.glob('Val/*augmented*.png')
    for element in val_augmented:
        if element in val_all:
            val_all.remove(element)
    tiles = val_all
    for f in tiles:
        os.remove(f)


def clean_augmented ():
    augmented_files = glob.glob('Train/*augmented*.png')
    for f in augmented_files:
        os.remove(f)

    augmented_files = glob.glob('Val/*augmented*.png')
    for f in augmented_files:
        os.remove(f)

#######################################################################
# Deep Forest: PREDICT
#######################################################################

# # # Predict image
# # imgDF = model.predict_tile(r"C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Agisoft\A3\Ortho test4.tif",return_plot=True,patch_overlap=0.25,patch_size=1500)

# # # Output bounding box locations/dimensions
# # boxes = model.predict_tile(r"C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Agisoft\A3\Ortho test4.tif",return_plot=False,patch_overlap=0.25,patch_size=1500)

# # Predict image
# imgDF, boxes = m.predict_tile(r"Agisoft\Ortho\Ortho cropped.tif",return_plot=True,patch_overlap=0.25,patch_size=1100)

# plt.imshow(imgDF[:,:,::-1])
# fig= plt.gcf()
# fig.set_size_inches(18.5*3, 10.5*3)



#######################################################################
# Local Maxima: Find
#######################################################################

def lm (dtm, distance, min_height, offset, max_peaks):
    """Searches the DTM for the local maxima

    Returns:
        coordinates and height of local maximas as a pandas dataframe
    """

    # raster = r"C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Agisoft\A3\DEM test3.tif"
    raster = "Agisoft/DEM/" + dtm
    with rasterio.open(raster) as source:
        img = source.read(1) 


    # locate max peak
    coordinates = peak_local_max(img, min_distance=distance, exclude_border=0, num_peaks=max_peaks, threshold_abs=min_height, p_norm=2)
    X=coordinates[:, 1]
    y=coordinates[:, 0]

    # create a data frame
    df = pd.DataFrame({'X':X, 'Y':y})

    # count seedlings
    count = df['X'].count()
    print('Total counted seedlings : {i}'.format(i = count))

    # Add height to the df
    df['Height']=0.0000

    temp=0.00
    for i in range(len(df)):
        temp = img[df['Y'][i]][df['X'][i]]
        temp=temp-offset
        # df['Height'][i] =temp
        df.loc[i, 'Height'] = temp

    # # Set threshold for heights
    # height_threshold=min_height
    # df=df[df.Height>height_threshold]

    # count viable seedlings remaining after threshold
    countV = df['X'].count()
    print('Total viable seedlings : {i}'.format(i = countV))

    return df


#######################################################################
# Assign nearest local maxima to Box
#######################################################################
def assign_lm (boxes, df_lm, max=False):
    """Assigns local max to bounding box based on euclid distance (box centroid to lm)

    Returns:
        boxes dataframe with assigned sampled height
    """
    if max == False:
        # Calculate box centroids
        Distance=[]
        boxes['xc']=boxes.xmin + (boxes.xmax-boxes.xmin)/2
        boxes['yc']=boxes.ymin + (boxes.ymax-boxes.ymin)/2
        boxes['Lmax_X']=np.nan
        boxes['Lmax_Y']=np.nan
        boxes['Height']=np.nan

        for i in range(len(boxes)):

            Distance = df_lm[(df_lm.X > (boxes.loc[i,'xmin']) ) & (df_lm.X < (boxes.loc[i,'xmax'])) & (df_lm.Y > (boxes.loc[i,'ymin'])) & (df_lm.Y < (boxes.loc[i,'ymax']))]

            if Distance.shape[0] > 1:
                for j in Distance.index:
                    Distance.loc[j, 'Euclid'] = dist( [boxes.loc[i,'xc'],boxes.loc[i,'yc']] , [Distance.loc[j,'X'],Distance.loc[j,'Y']] )
                    # print(j)
                temp = Distance[Distance.Euclid==Distance.Euclid.min()].to_numpy()
                boxes.loc[i,'Lmax_X'] = temp[0][0]
                boxes.loc[i,'Lmax_Y'] = temp[0][1]
                boxes.loc[i,'Height'] = temp[0][2]
            elif Distance.shape[0] == 1:
                temp=Distance.to_numpy()
                boxes.loc[i,'Lmax_X'] = temp[0][0]
                boxes.loc[i,'Lmax_Y'] = temp[0][1]
                boxes.loc[i,'Height'] = temp[0][2]


        # count viable seedlings
        countNB = boxes[pd.isna(boxes.Height) == True].shape[0]
        print('Number of trees without an assigned height: {i}'.format(i = countNB))
    else:
        # Use box maximums
        # Calculate box centroids
        Distance=[]
        boxes['xc']=boxes.xmin + (boxes.xmax-boxes.xmin)/2
        boxes['yc']=boxes.ymin + (boxes.ymax-boxes.ymin)/2
        # boxes['Lmax_X']=np.nan
        # boxes['Lmax_Y']=np.nan
        boxes['Height']=np.nan

        for i in range(len(boxes)):

            Distance = df_lm[(df_lm.X > (boxes.loc[i,'xmin']) ) & (df_lm.X < (boxes.loc[i,'xmax'])) & (df_lm.Y > (boxes.loc[i,'ymin'])) & (df_lm.Y < (boxes.loc[i,'ymax']))]

            boxes.loc[i,'Height'] = Distance.Height.max()


        # count viable seedlings
        countNB = boxes[pd.isna(boxes.Height) == True].shape[0]
        print('Number of trees without an assigned height: {i}'.format(i = countNB))
    return boxes


#######################################################################
# # Assign MAX local maxima to Box
#######################################################################
def assign_lm_max (boxes, df_lm):
    """Assigns max local max to bounding box

    Returns:
        boxes dataframe with assigned sampled height
    """
    # Calculate box centroids
    Distance=[]
    boxes['xc']=boxes.xmin + (boxes.xmax-boxes.xmin)/2
    boxes['yc']=boxes.ymin + (boxes.ymax-boxes.ymin)/2
    # boxes['Lmax_X']=np.nan
    # boxes['Lmax_Y']=np.nan
    boxes['Height']=np.nan

    for i in range(len(boxes)):

        Distance = df_lm[(df_lm.X > (boxes.loc[i,'xmin']) ) & (df_lm.X < (boxes.loc[i,'xmax'])) & (df_lm.Y > (boxes.loc[i,'ymin'])) & (df_lm.Y < (boxes.loc[i,'ymax']))]

        boxes.loc[i,'Height'] = Distance.Height.max()


    # count viable seedlings
    countNB = boxes[pd.isna(boxes.Height) == True].shape[0]
    print('Number of trees without an assigned height: {i}'.format(i = countNB))

    return boxes


#######################################################################
# Segment image into grid (CROPPED image only)
#######################################################################

def segment (imgDF, boxes):
    """Segment tray to find plug locations
        Assign index number to plug location

    Returns:
        coordinates of plug centroids as pandas dataframe
    """
    df_grid=pd.DataFrame(columns=['row', 'col', 'number', 'x_coord', 'y_coord'])

    trayEdge = 45
    imgSize= imgDF.shape
    imgYmax = imgSize[0]
    imgXmax = imgSize[1]
    
    xstart = boxes.xmin.min()
    xend = imgXmax-boxes.xmax.max()
    ystart = boxes.ymin.min()
    yend = imgYmax-boxes.xmax.max()

    if xstart < trayEdge:
        xstart = trayEdge
    if xend < trayEdge:
        xend = trayEdge
    if ystart < trayEdge:
        ystart = trayEdge
    if yend < trayEdge:
        yend = trayEdge

    
    row_shift=(imgYmax-ystart-yend)/7
    col_shift=(imgXmax-xstart-xend)/14
    # x_shift = trayEdge + col_shift*0.5
    y_shift = ystart-5 + row_shift*0.5
    s=0

    # xstart=xstart-15
    # create grid for seedling tray positions
    for row in range(7):
        x_shift = xstart + col_shift*0.5
        for col in range (14):
            df_grid.loc[s, ['row', 'col', 'number', 'x_coord', 'y_coord']] = row, col, s+1, x_shift, y_shift
            x_shift += col_shift
            s += 1
        y_shift += row_shift

    return df_grid


#######################################################################
# Index seedlings
#######################################################################

def index (boxes, df_grid):
    """Index seedlings
        Assign box to plug location and assign index number to box
        Assign index number to plug location

    Returns:
        boxes dataframe with index numbers
    """
    # Assign position to seedling
    compdf=pd.DataFrame(columns=['Distance'])
    for i in range(boxes.shape[0]):
        for j in range(df_grid.shape[0]):
            d = dist([boxes.loc[i,'xc'], boxes.loc[i,'yc']], [df_grid.loc[j,'x_coord'],df_grid.loc[j,'y_coord']])
            compdf.loc[j, 'Distance'] = d
        
        id=compdf[compdf.Distance == compdf.Distance.min()].index[0].astype(int)
        boxes.loc[i, ['position','Distance_to_pos']] = id.astype(int), compdf.loc[id, 'Distance']

    # Code to remove duplicates
    temp = boxes[boxes['position'].duplicated(keep= False)]

    # # Unassign the furthest duplicate from the tray position
    # # Filters out instances assigned to the same position and replaces furthest instance with NaN
    # for pos in temp.position:
    #     subset = temp[temp.position == pos]
    #     subset.loc[subset['Distance_to_pos'] == subset['Distance_to_pos'].max(), 'position'] = np.nan
    #     for i in subset.index:
    #         boxes.loc[i] = subset.loc[i]

    # Reassign box instances when positions have been duplicated
    for pos in temp.position.unique().astype(int):

        # Get instances with duplicated positions
        subset = temp[temp.position == pos]

        # Get instances to reassign
        reassign = subset.loc[subset['Distance_to_pos'] == subset['Distance_to_pos'].max()]

        # Get incorrectly assigned position
        incorrect_pos = reassign.position.values[0].astype(int)

        # Get available neighbours
        neigh = neighbours(incorrect_pos)
        taken_pos = list(boxes.position.unique().astype(int))
        for element in taken_pos:
            if element in neigh:
                neigh.remove(element)
        if len(neigh) != 0:
            # Calculare distance to available neighbours
            Re_dist=pd.DataFrame(columns=['position','Distance'])
            for idx, n in enumerate(neigh):
                rd = dist([reassign.xc, reassign.yc], [df_grid.loc[n,'x_coord'],df_grid.loc[n,'y_coord']])
                Re_dist.loc[idx, ['position','Distance']] = n, rd

            # Assign new position to instance
            reassign.loc[:,'position'] = Re_dist[Re_dist.Distance == Re_dist.Distance.min()].position.values[0]
        else:
            reassign.loc[:,'position'] = np.nan
        
        # Update boxes dataframe
        for i in reassign.index:
            boxes.loc[i] = reassign.loc[i]

    return boxes


#######################################################################
# Plot true vs predicted Height and calculate prediction error
#######################################################################
def th(boxes, path):

    # path='Heights/' + height

    true_height=pd.read_excel(path, index_col=0)

    dfFinal= pd.merge(boxes,true_height,left_on = 'position', right_on='number')
    # dfFinal= pd.merge(true_height,boxes,left_on = 'number', right_on='position', how='outer')
    dfFinal['errorH'] = dfFinal['Height'] - dfFinal['true_height']
    dfFinal=dfFinal.sort_values(by='position', ascending=True).reset_index(drop=True)

    # # Only evaluate seedlings with predicted height
    # dfFinalValid=dfFinal[dfFinal.Height.isna() == False].copy()

    # plt.plot(dfFinalValid.position, dfFinalValid.Height)
    # plt.plot(dfFinalValid.position, dfFinalValid.true_height)
    # fig= plt.gcf()
    # fig.set_size_inches(18.5*3, 10.5)

    # # Calculate Height prediction error
    # error=abs(dfFinalValid.errorH)
    # error.describe()
    # dfFinalValid.errorH.describe()

    return dfFinal

#######################################################################
# Extract seedling crops and save
#######################################################################

def seedling_extract (dfFinal, imgO, seedling_dir=None, tuning=True):

    seedling_dir='Seedling_Images/'
    # Get index positions of boxes
    BoxWindows = dfFinal[['ymin','ymax', 'xmin','xmax']]
    BoxWindows = BoxWindows.astype(int)
    BoxWindows = BoxWindows.values.tolist()
    BoxPos=dfFinal['position'].astype(int).tolist()

    # Extract seedling arrays
    cropB =[]
    for BoxWindow in BoxWindows:
        seedling = imgO[BoxWindow[0]:BoxWindow[1], BoxWindow[2]:BoxWindow[3]]
        cropB.append(seedling)

    if tuning == False:
        # Save cropped images
        for idxb, cB in enumerate(cropB):
            preprocess.save_crop(seedling_dir, dfFinal.Tray[idxb]+'_seedling', BoxPos[idxb], cB)



    return BoxPos, cropB

#######################################################################
# Extract seedling features
#######################################################################

def seedling_features (BoxPos, dfFinal, cropB):

    seedlings = pd.DataFrame(dfFinal.Tray.values, columns=['tray'])
    seedlings['position'] = BoxPos
    seedlings['boxWidth'] = dfFinal.xmax-dfFinal.xmin
    seedlings['boxHeight'] = dfFinal.ymax-dfFinal.ymin
    seedlings['boxArea'] = seedlings.boxWidth * seedlings.boxHeight

    for i, cS in enumerate(cropB):
        b,g,r = cv2.split(cS)
        seedlings.loc[i, ['green','blue','red']] = np.mean(g), np.mean(b), np.mean(r)

    seedlings['SampledHeight'] = dfFinal.Height

    # Gather information from neighbouring seedlings
    top=np.linspace(0,13,14)
    bottom=np.linspace(84,97,14)
    left=np.linspace(0,84,7)
    right=np.linspace(13,97,7)
    neighbourList=[]

    for i, pos in enumerate(seedlings.position):
        edge=0
        neighbours=[]
        if (pos in top) & ((pos in left)==False) & ((pos in right)==False):
            neighbours=[pos-1, pos+1, pos+13, pos+14, pos+15]
            edge=1
        elif (pos in bottom) & ((pos in left)==False) & ((pos in right)==False):
            neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1]
            edge=1
        elif (pos in left) & ((pos in top)==False) & ((pos in bottom)==False):
            neighbours=[pos-14, pos-13, pos+1, pos+14, pos+15]
            edge=1
        elif (pos in right) & ((pos in top)==False) & ((pos in bottom)==False):
            neighbours=[pos-15, pos-14, pos-1, pos+13, pos+14]
            edge=1
        elif pos == 0:
            neighbours=[pos+1,pos+14, pos+15]
            edge=1
        elif pos == 13:
            neighbours=[pos-1, pos+13, pos+14]
            edge=1
        elif pos == 84:
            neighbours=[pos-14, pos-13, pos+1]
            edge=1
        elif pos == 97:
            neighbours=[pos-15, pos-14, pos-1]
            edge=1
        else:
            neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1, pos+13, pos+14, pos+15]

        neighbourList.append(neighbours) 
        seedlings.at[i, 'neighbour_meanHeight'] = np.mean(seedlings.loc[seedlings['position'].isin(neighbours)].SampledHeight)#.astype(float)
        seedlings.at[i, 'neighbour_maxHeight'] = seedlings.loc[seedlings['position'].isin(neighbours)].SampledHeight.max()
        seedlings.loc[i, 'edge'] = edge

    # Impute missing samples with mean
    SampledHeight_mean = seedlings['SampledHeight'].mean()
    for row in seedlings[seedlings['SampledHeight'].isnull()==True].index:
        seedlings.loc[row, 'SampledHeight'] = SampledHeight_mean
    # Create new features
    seedlings['neighbour_meanHeight_Diff'] = seedlings['neighbour_meanHeight'] - seedlings['SampledHeight']
    seedlings['neighbour_maxHeight_Diff'] = seedlings['neighbour_maxHeight'] - seedlings['SampledHeight']

    seedlings= pd.merge(seedlings,dfFinal[['position','true_height']],left_on = 'position', right_on='position' )
    seedlings['errorH'] = seedlings['SampledHeight'] - seedlings['true_height']

    return seedlings



#######################################################################
# Get neighbours
#######################################################################

def neighbours (position):

    # Gather information from neighbouring seedlings
    top=np.linspace(0,13,14)
    bottom=np.linspace(84,97,14)
    left=np.linspace(0,84,7)
    right=np.linspace(13,97,7)
    neighbourList=[]

    if isinstance(position, list):
        for i, pos in enumerate(position):
            edge=0
            neighbours=[]
            if (pos in top) & ((pos in left)==False) & ((pos in right)==False):
                neighbours=[pos-1, pos+1, pos+13, pos+14, pos+15]
                edge=1
            elif (pos in bottom) & ((pos in left)==False) & ((pos in right)==False):
                neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1]
                edge=1
            elif (pos in left) & ((pos in top)==False) & ((pos in bottom)==False):
                neighbours=[pos-14, pos-13, pos+1, pos+14, pos+15]
                edge=1
            elif (pos in right) & ((pos in top)==False) & ((pos in bottom)==False):
                neighbours=[pos-15, pos-14, pos-1, pos+13, pos+14]
                edge=1
            elif pos == 0:
                neighbours=[pos+1,pos+14, pos+15]
                edge=1
            elif pos == 13:
                neighbours=[pos-1, pos+13, pos+14]
                edge=1
            elif pos == 84:
                neighbours=[pos-14, pos-13, pos+1]
                edge=1
            elif pos == 97:
                neighbours=[pos-15, pos-14, pos-1]
                edge=1
            else:
                neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1, pos+13, pos+14, pos+15]

            neighbourList.append(neighbours) 
        return neighbourList
    else:
        pos = position
        if (pos in top) & ((pos in left)==False) & ((pos in right)==False):
            neighbours=[pos-1, pos+1, pos+13, pos+14, pos+15]
            edge=1
        elif (pos in bottom) & ((pos in left)==False) & ((pos in right)==False):
            neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1]
            edge=1
        elif (pos in left) & ((pos in top)==False) & ((pos in bottom)==False):
            neighbours=[pos-14, pos-13, pos+1, pos+14, pos+15]
            edge=1
        elif (pos in right) & ((pos in top)==False) & ((pos in bottom)==False):
            neighbours=[pos-15, pos-14, pos-1, pos+13, pos+14]
            edge=1
        elif pos == 0:
            neighbours=[pos+1,pos+14, pos+15]
            edge=1
        elif pos == 13:
            neighbours=[pos-1, pos+13, pos+14]
            edge=1
        elif pos == 84:
            neighbours=[pos-14, pos-13, pos+1]
            edge=1
        elif pos == 97:
            neighbours=[pos-15, pos-14, pos-1]
            edge=1
        else:
            neighbours=[pos-15, pos-14, pos-13, pos-1, pos+1, pos+13, pos+14, pos+15]
        return neighbours



#######################################################################
# Combine list of DFs in single DF
#######################################################################

def joinl (lst):
    """Combines a list of dataframes into single dataframe
    Returns:
        Combined pandas dataframe
    """
    if isinstance(lst, list):
        for i, l in enumerate(lst):
            if i == 0:
                comb = l
            else:
                frames =[comb, l]
                comb = pd.concat(frames)
        return comb
    else:
        return lst

#######################################################################
# Calculate scores for regression models
#######################################################################
def regression_scores(y_true, y_pred, model, features=None, feature_list_id=None):

    score_df = pd.DataFrame(columns=['model', 'mae', 'rmse', 'mape', 'r2', 'features', 'feature_list_idx'])

    score_df['features'] = score_df['features'].astype(object)
    score_df.loc[0,'model'] = model
    score_df.loc[0,'mae'] = mean_absolute_error(y_true, y_pred)
    score_df.loc[0,'rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    score_df.loc[0,'mape'] = mean_absolute_percentage_error(y_true, y_pred)
    score_df.loc[0,'r2'] = r2_score(y_true, y_pred)
    
    if features != None or feature_list_id != None:
        score_df.loc[0,'feature_list_idx'] = feature_list_id
        score_df.loc[0,'features'] = features

    return score_df

#%%################################################################################################
# INITIAL SETUP FOR TUNING
################################################################################################

# parameters
Train_dir = 'Train/'
Val_dir = 'Val/'
Test_dir = 'Test/'
seedling_dir='Seedling_Images/'
images = glob.glob(r"Agisoft\Ortho\*.tif")
cvat_file = 'annotations.zip'

validation_split = 0.25
test_split = 0.25
train_split = 1-(validation_split+test_split)


#############################################
#  UNZIP CVAT FILE
annotations_original = unzip(cvat_file, 'Agisoft/Ortho/','annotations_original.csv' )


#############################################
#  TRAIN, VALIDATION, TEST SPLIT
train_annotations, val_annotations, test_annotations = train_val_test_split(annotations_original, validation_split, test_split)




# #%%
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################

# windowsO = tr_windowsO
# annotations = train_annotations
# sel_annotations_df = pd.DataFrame()
# # def tile_annotations (windowsO, annotations):

# sel_annotations=[]

# for idx, windows in enumerate (windowsO):
#     for i in range(len(windows)):
#         crop_annotations=preprocess.select_annotations(annotations[annotations.image_path == annotations.image_path.unique()[idx]], windows, i)
#         sel_annotations.append(crop_annotations)


# #%%
#     for j in range(len(sel_annotations)):
#         if j == 0:
#             sel_annotations_df = sel_annotations[j]
#         else:
#             frames=[sel_annotations_df, sel_annotations[j]]
#             sel_annotations_df = pd.concat(frames)


# #%%

# idx=1
# j=0

# # sel_annotations_df = sel_annotations[j]

# # frames=[sel_annotations_df, sel_annotations[j]]
# # sel_annotations_df = pd.concat(frames)


# # t_annotations = sel_annotations_df

# #%%
# sel_annotations
# # t_annotations

# # range(len(windows))

# # tr_tile_annotations[(tr_tile_annotations.image_path=='Tray5_ortho_0.png')&(tr_tile_annotations.xmin==61)]

# # sel_annotations_df[(sel_annotations_df.image_path=='Tray5_ortho_0.png')&(sel_annotations_df.xmin==61)]
# # sel_annotations_df
# #%%
#     if idx == 0:
#         t_annotations = sel_annotations_df
#     else:
#         frames=[t_annotations, sel_annotations_df]
#         t_annotations = pd.concat(frames)

# # return t_annotations



# # val_tile_annotations[val_tile_annotations.xmin==41]



# # tr_tile_annotations[(tr_tile_annotations.image_path=='Tray5_ortho_0.png')&(tr_tile_annotations.xmin==61)]
# # train_annotations
# # val_annotations


# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################
# #%%




#############################################
#  CREATE TILES

tileSize = 1000
tileOverlap = 0.25
Aug_images_required_train=1
Augmentation='none'
min_vis=0.3

clean_tiles()

# train data
tr_imagesO=[]
tr_windowsO=[]
for train_image in train_annotations.image_path.unique():
    imgO, windows = tiles (train_image, tileSize, tileOverlap,Train_dir)
    tr_imagesO.append(imgO)
    tr_windowsO.append(windows)

# val data
val_imagesO=[]
val_windowsO=[]
for val_image in val_annotations.image_path.unique():
    imgO, windows = tiles (val_image, tileSize, tileOverlap,Val_dir)
    val_imagesO.append(imgO)
    val_windowsO.append(windows)

#############################################
#  GET TILE ANNOTATIONS

# train data
tr_tile_annotations = tile_annotations (tr_windowsO, train_annotations)

# val data
val_tile_annotations = tile_annotations (val_windowsO, val_annotations)


#############################################
#  AUGMENT IMAGES AND GET ANNOTATIONS

clean_augmented()

# train data
tr_augmented_annotations = augment(tr_tile_annotations, Aug_images_required_train, Train_dir,Augmentation,min_vis) 

# val data
val_augmented_annotations = augment(val_tile_annotations, round(Aug_images_required_train*(validation_split/train_split)), Val_dir,Augmentation,min_vis) 



#############################################
#  SAVE ANNOTATIONS

# train data
train_anno_final = tr_augmented_annotations.copy()
annotations_file = save_annotations(train_anno_final, Train_dir)

# val data
val_anno_final = val_augmented_annotations.copy()
validation_file = save_annotations(val_anno_final, Val_dir)

# test data
clean_test()
test_FileNames = glob.glob(r'Agisoft\Ortho\test\*')
for name in test_FileNames:
    shutil.copy(name, Test_dir)
test_file = Test_dir + 'test.csv'


#%%#############################################
#  Hyperparameter tuning
#############################################


# DeepForest config with default settings
nms_threshold = 0.05
score_threshold = 0.1
# epoch=15
validation_iou_threshold = 0.4
# lr_schedules = ['default' ,'StepLR', 'exponential']
# optimisers = ['sgd', 'Adadelta', 'Adam', 'Rprop']
# train_learning_rates = [0.1, 0.01, 0.001, 0.0001]#[0.0001]
# batch_sizes = [1,2,3,4,5,6]
#%%
#########################
# Just for log files
epoch=4
lr_schedules =          ['default']# ,'StepLR']
optimisers =     ['sgd']#        ['Adam']
train_learning_rates = [0.001,0.001,0.001,0.001]#[0.1, 0.01, 0.001, 0.0001]
batch_sizes =  [4]#         [1,2,3,4]


#########################

results_df = pd.DataFrame(columns=['batch_size','train_learn_rate', 'learn_rate', 'optimiser', 'box_precision', 'box_recall', 'box_f1', 'class_precision', 'class_recall', 'class_f1','avg_precision','avg_recall','avg_f1', 'miou','train_time'])
idx = 0
for train_learning_rate in train_learning_rates:
    for batch_size in batch_sizes:
        for learn_rate in lr_schedules:
            for optimiser in optimisers:

                file_name = 'Hyp__'+'opti='+optimiser+'_'+'lr_init='+str(train_learning_rate)+'_'+'lr='+learn_rate+'_'+'bS='+str(batch_size)
                
                # Configure model
                m = config (annotations_file, validation_file, batch_size, nms_threshold, score_threshold, epoch, train_learning_rate, validation_iou_threshold,optimiser,learn_rate,file_name)

                # Fit model
                start_time = time.time()
                m.trainer.fit(m)
                train_time = time.time() - start_time
                

                model_path = 'DF_models/Hyper_testing/'+file_name+'.pt'
                torch.save(m.model.state_dict(),model_path)
                


                # Evaluate model
                save_dir = 'C:/Users/camer/Documents/DF_logs/Hyper_tuning/'
                results = DFeval(m, validation_file, validation_iou_threshold, save_dir)

                # Save results
                results_df.loc[idx, 'train_time'] = train_time
                results_df.loc[idx, 'learn_rate'] = learn_rate
                results_df.loc[idx, 'train_learn_rate'] = train_learning_rate
                results_df.loc[idx, 'optimiser'] = optimiser
                results_df.loc[idx, 'batch_size'] = batch_size
                results_df.loc[idx, 'box_precision'] = results['box_precision']
                results_df.loc[idx, 'box_recall'] = results['box_recall']
                true_positive = results['results']['match'].sum()
                results_df.loc[idx, 'class_recall'] = true_positive / val_anno_final.shape[0]
                results_df.loc[idx, 'class_precision'] = true_positive / results['results'].shape[0]
                results_df.loc[idx, 'miou'] = results['results']['IoU'].mean()

                if results_df.loc[idx, 'box_recall'] + results_df.loc[idx, 'box_precision'] == 0: results_df.loc[idx, 'box_f1'] = 0
                else: results_df.loc[idx, 'box_f1'] = 2*((results_df.loc[idx, 'box_recall']*results_df.loc[idx, 'box_precision'])/(results_df.loc[idx, 'box_recall']+results_df.loc[idx, 'box_precision']))

                if results_df.loc[idx, 'class_recall'] + results_df.loc[idx, 'class_precision'] == 0: results_df.loc[idx, 'class_f1'] = 0
                else: results_df.loc[idx, 'class_f1'] = 2*((results_df.loc[idx, 'class_recall']*results_df.loc[idx, 'class_precision'])/(results_df.loc[idx, 'class_recall']+results_df.loc[idx, 'class_precision']))

                results_df.loc[idx, 'avg_precision'] = (results_df.loc[idx, 'box_precision'] + results_df.loc[idx, 'class_precision'])*0.5
                results_df.loc[idx, 'avg_recall'] = (results_df.loc[idx, 'box_recall'] + results_df.loc[idx, 'class_recall'])*0.5

                if results_df.loc[idx, 'avg_recall'] + results_df.loc[idx, 'avg_precision'] == 0: results_df.loc[idx, 'class_f1'] = 0
                else: results_df.loc[idx, 'avg_f1'] = 2*((results_df.loc[idx, 'avg_recall']*results_df.loc[idx, 'avg_precision'])/(results_df.loc[idx, 'avg_recall']+results_df.loc[idx, 'avg_precision']))

                idx += 1

                # clear cuda memory
                torch.cuda.empty_cache()
                del m
                gc.collect()

# results_df.to_csv('Results/DeepForest_results/Hyperparameter_tuning_results_lr_0_0001.csv', index=False)

#%%
results_df
# epoch15=results_df.copy()
# epoch15.to_csv('Results/DeepForest_results/Hyperparameter_tuning_results_epoch_15.csv', index=False)

#%%
results_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results.csv',delimiter=',')

results_df=results_df.drop(columns='train_learning_rate')
#%%

results_df=pd.DataFrame()
first_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results.csv',delimiter=',')
first_df=first_df.drop(columns='train_learning_rate')
first_df=first_df[first_df.optimiser != 'reduce_on_plat']

second_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results_lr_0_0001.csv',delimiter=',')

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

results_df = first_df.append(second_df)

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)
#%%#############################################
# Investigate results



#%%

# results_df=results_df.append(first_df)

# results_df[results_df.avg_f1 >=0.96]

results_df.sort_values('avg_f1', ascending=False).head(5)

#%%

# first_halfb = first_half.copy()
# true_positive/val_anno_final.shape[0]
# second_halfb=second_half.copy()
# validation_file
# results_df.to_csv('Results/DeepForest_results/Hyperparameter_tuning_results_first_half.csv', index=False)
# second_half
# results_df.info()
# first_half.train_learn_rate.unique()
# first_half = first_half.drop(labels=[192,193], axis=0)
# tester=results_df.copy
# results_df = first_half.append(second_half)
# results_df[results_df.class_recall == results_df.class_recall.max()]
# results_df.drop(index=['train_learning_rate'], axis=1)
# results_df

# results_df[results_df.avg_f1 == results_df.avg_f1.max()]
# results_df[results_df.avg_f1 >=0.96].sort_values('avg_f1',ascending=False)

# results_df.sort_values('avg_f1',ascending=False).head(10)
# results_df[results_df.box_f1 == results_df.box_f1.max()]
# results_df.train_learn_rate.value_counts()
#%%
# import seaborn as sns
features=['box_precision','class_precision','box_recall','class_recall','box_f1','class_f1']
plot_data1=results_df.loc[:,features].copy()
#%%
plot_data1.astype(float)
#%%
#################################################################################################################################################################################################################################
# PLOTS
#################################################################################################################################################################################################################################

# Import saved CSV files

results_df=pd.DataFrame()
first_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results.csv',delimiter=',')
first_df=first_df.drop(columns='train_learning_rate')
first_df=first_df[first_df.optimiser != 'reduce_on_plat']

second_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results_lr_0_0001.csv',delimiter=',')

results_df = first_df.append(second_df)

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

results_df_valid



#%%
##################################
# Plot all
import seaborn as sns

plot_data=[results_df['box_precision'].astype(float),results_df['class_precision'].astype(float),results_df['box_recall'].astype(float),results_df['class_recall'].astype(float),results_df['box_f1'].astype(float),results_df['class_f1'].astype(float)]

labels=['Box precision','Class precision','Box recall','Class recall','Box f1','Class f1']
# plot_data_valid=plot_data1.dropna(axis=0)


my_pal = {'box_precision': 'b','class_precision':'b','box_recall':'g','class_recall':'b','box_f1':'m','class_f1':'m'}


ax = sns.violinplot(data=plot_data,cut=0)#, palette=my_pal)

for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)


sns.despine(offset=10, trim=True, bottom=True)
# sns.set_context("paper")
sns.set_theme(style="ticks")
ax.tick_params(bottom=False)
ax.set_xticklabels(labels)
fig = plt.gcf()
fig.set_size_inches(18.5*0.6, 10.5*0.6)
plt.show()

#%%
#########################################
# Plot averages

# vertical
import seaborn as sns
# plt.rc(usetex=True)
plt.rc('pgf', texsystem='pdflatex')
# plt.rc(usetex=True)
features=['avg_precision','avg_precision']
plot_data=[results_df['avg_precision'].astype(float),results_df['avg_precision'].astype(float)]
# plot_data=plot_data.dropna(axis=0)

# labels=['Avgerage \nprecision','Avgerage \nrecall']
labels=['Avg precision','Avg recall']
my_pal = {'avg_precision': 'b','avg_precision':'b'}

ax = sns.violinplot(data=plot_data, cut=0)#, palette=my_pal)
for violin, alpha in zip(ax.collections[::2], [0.8,0.8]):
    violin.set_alpha(alpha)


sns.despine(offset=10, trim=True, bottom=True)
# sns.set_context("paper")
sns.set_theme(style="ticks")
ax.tick_params(bottom=False)
ax.set_xticklabels(labels)
fig = plt.gcf()
fig.set_size_inches(4,6)
#%%
#########################################
# Plot averages - VERTICAL

# # np.isnan(results_df_valid['avg_precision']).any()
# results_df_valid[np.isinf(results_df_valid['miou'])==True]


results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

# csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
features=['avg_f1','miou']
plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
labels=['Avg F1', 'Mean IOU']
ax = sns.violinplot(data=plot_data, orient='v')#, palette=my_pal)
for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_ylim(-0.01, 1.001)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)
ax.set_xticklabels(labels)
ticklbls=ax.get_xticklabels(which='both')
for y in ticklbls:
    y.set_ha('left')
lbl_size=14

ax.get_xaxis().set_tick_params(labelsize=lbl_size, direction='out')#,pad=65)
ax.get_yaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(4,8)

image_name='Hypers_avgF1_avgIOU.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=600)
#%%
#########################################
# Plot averages - HORIZONTAL

# # np.isnan(results_df_valid['avg_precision']).any()
# results_df_valid[np.isinf(results_df_valid['miou'])==True]
results_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results_combined.csv',delimiter=',')
# results_df
#%%

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

# csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
features=['avg_f1','miou']
plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
labels=['avgF1', 'mIOU']


plt.figure(dpi=1200)
ax = sns.violinplot(data=plot_data, orient='h')#, palette=my_pal)
for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_xlim(0, 1.002)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)
ax.set_yticklabels(labels)
ticklbls=ax.get_yticklabels(which='both')
for x in ticklbls:
    x.set_ha('left')
lbl_size=15

ax.set_xlabel('Score',fontsize=lbl_size,labelpad=10)

ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=75)

ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(8,4)

image_name='Hypers_avgF1_avgIOU.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

# features=['avg_precision','avg_precision','avg_f1','miou']
# plot_data=[results_df_valid['avg_precision'].astype(float),results_df_valid['avg_precision'].astype(float),results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]

# labels=['Avgerage \nprecision','Avgerage \nrecall']
# labels=['Mean precision','Mean recall', 'Mean f1', 'Avg IOU']
# my_pal = {'avg_precision': 'b','avg_precision':'b'}

# ax = sns.violinplot(data=plot_data, cut=0, orient='h')#, palette=my_pal)
# sns.set_context("paper")
# yax = ax.get_yaxis()
# pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
# yax.set_tick_params(pad=pad)

# plt.figure(figsize=(15, 3))

# plt.show()


#%%
#########################################
# PLOT OPTIMISERS AVERAGES - HORIZONTAL

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)


# features=['$vg_f1','miou']
# plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
labels=['SGD', 'AdaDelta', 'Adam', 'Rprop']
lbl_size=20
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)
ax = sns.violinplot(x=results_df.avg_f1, y=results_df.optimiser, data=results_df)#, palette=my_pal)

for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_xlim(0, 1.002)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

ax.set(ylabel=None)
ax.set_xlabel('avgF1 score',fontsize=lbl_size,labelpad=10)
ax.set_ylabel('Optimiser',fontsize=lbl_size,labelpad=10)

ax.set_yticklabels(labels)
# ticklbls=ax.get_yticklabels(which='both')

# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(12,6)

image_name='Hypers_Optimisers.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%
#########################################
# PLOT INITIAL LEARNING RATE AVERAGES - HORIZONTAL

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

results_df_valid_sorted = results_df_valid.sort_values('train_learn_rate',ascending=False).copy()


# features=['$vg_f1','miou']
# plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
# labels=['0.001', '0.01', '0.1']
lbl_size=20
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)
ax = sns.violinplot(x=results_df.avg_f1, y=results_df.train_learn_rate, data=results_df, orient='h')#, palette=my_pal)

for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_xlim(0, 1.002)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

# ax.set(ylabel=None)
ax.set_xlabel('avgF1 score',fontsize=lbl_size, labelpad=0)
# ax.set_xlabel('mIOU',fontsize=lbl_size, labelpad=10)
ax.set_ylabel('Initial learning rate',fontsize=lbl_size, labelpad=10)

# ax.set_yticklabels(labels)
ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(12,6)

image_name='Hypers_Initial_lr.png'
# image_name='Hypers_Initial_lr_miou.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%
#########################################
# PLOT SCHEDULE AVERAGES - HORIZONTAL

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

results_df_valid_sorted = results_df_valid.sort_values('learn_rate',ascending=False).copy()
results_df_valid_schedule=results_df_valid[results_df_valid.learn_rate !='reduce_on_plat']

# features=['$vg_f1','miou']
# plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
labels=['RoP', 'Stepped', 'Exponential']
lbl_size=20
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)
ax = sns.violinplot(x=results_df.avg_f1, y=results_df.learn_rate, data=results_df, orient='h')#, palette=my_pal)

for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_xlim(0, 1.002)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

# ax.set(ylabel=None)
ax.set_xlabel('avgF1 score',fontsize=lbl_size, labelpad=10)
ax.set_ylabel('Schedule',fontsize=lbl_size, labelpad=0)

ax.set_yticklabels(labels)
ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(12,6)

image_name='Hypers_Initial_lr_schedule.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')


#%%
#########################################
# PLOT BATCH AVERAGES - HORIZONTAL

results_df=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results_combined.csv',delimiter=',')

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

lbl_size=22
# features=['$vg_f1','miou']
# plot_data=[results_df_valid['avg_f1'].astype(float),results_df_valid['miou'].astype(float)]
# labelsB=['Batch size=1', 'Batch size=2', 'Batch size=3', 'Batch size=4', 'Batch size=5', 'Batch size=6']
labelsB=['1', '2', '3', '4', '5', '6']


plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)
ax = sns.violinplot(x=results_df.avg_f1, y=results_df.batch_size, data=results_df, orient='h')#, palette=my_pal)

for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax.set_xlim(0, 1.001)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

ax.set(xlabel=None)
ax.set_ylabel('Batch size',fontsize=lbl_size, labelpad=10)
ax.set_xlabel('avgF1 score',fontsize=lbl_size, labelpad=10)
ax.set_yticklabels(labelsB)
# ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')




ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(12,8)

image_name='Hypers_Batches.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%
#########################################
# PLOT HYPERPARAMETER AVERAGES - VERTICAL

results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
results_df_valid=results_df.dropna(axis=0)

# ax[0, 0].plot(range(10), 'r') #row=0, col=0
# ax[1, 0].plot(range(10), 'b') #row=1, col=0
# ax[0, 1].plot(range(10), 'g') #row=0, col=1
# ax[1, 1].plot(range(10), 'k') #row=1, col=1

labels_opti=['SGD', 'Adelta', 'Adam', 'Rprop']
labels_lr=['Reduce \non plateau', 'Step', 'Adam', 'Rprop']
plt.rcParams["font.family"] = "Times New Roman"
lbl_size=14

fig, ax =plt.subplots(1,2, sharey=True)


sns.set_theme(style="whitegrid")
sns.violinplot(y=results_df_valid.avg_f1,x=results_df_valid.optimiser, data=results_df_valid, ax=ax[0])
# plt.ylabel('Average F1 score',fontsize=lbl_size)
ax[0].set_ylabel('Average F1 score',fontsize=lbl_size)
ax[0].set_xlabel('Optimiser',fontsize=lbl_size)
ax[0].set_xticklabels(labels_opti)

for violin, alpha in zip(ax[0].collections[::2], [0.8,0.8,0.8,0.8]):
    violin.set_alpha(alpha)
ax[0].set_ylim(0, 1)
ax[0].get_xaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=10)
ax[0].get_yaxis().set_tick_params(labelsize=lbl_size)



sns.violinplot(y=results_df_valid.avg_f1,x=results_df_valid.learn_rate, data=results_df_valid, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].get_xaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=10)
ax[1].get_yaxis().set_tick_params(labelsize=lbl_size)



sns.despine(offset=10, trim=True, bottom=True, left=True)


# ax.tick_params(left=False)
# ax.set_xticklabels(labels)



# plt.xlabel('Optimiser',fontsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(10,6)

plt.show()


image_name='Hypers_Optimisers.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=600)



#%%
results_df[results_df.isna().any(axis=1)]
# results_df[results_df.isna()==True]
# results_df
# results_df.isna().any()

#%%
results_df.groupby('batch_size')['train_time'].mean()
# results_df.groupby('batch_size')['avg_f1'].mean()miou
# results_df['avg_f1'].median()
# 'batch_size','train_learn_rate', 'learn_rate', 'optimiser',train_time

# results_df_valid[np.isinf(results_df_valid['miou'])==True]
# results_df.is_inf().any()
# results_df.miou.max()

#%%

# custom_params = {"axes.spines.right": False, "axes.spines.bottom": False, "axes.spines.top": False,}
# sns.set_theme(style="ticks", rc=custom_params)

# sns.set_theme(style="whitegrid", rc=custom_params)
sns.set_theme(style="ticks")#, rc=custom_params)

ax = sns.violinplot(data=plot_data,cut=0)
# ax.set_xlabel('f')
sns.despine(offset=10, trim=True, bottom=True)
sns.set_context("paper")
# labels = ['A', 'B', 'C', 'D']
# set_axis_style(ax, labels)

ax.xaxis.set_tick_params(direction='out')

# ax.xaxis.set_ticks_position('bottom')
# ax.set_xticks()
ax.tick_params(bottom=False)
ax.set_xticklabels(features)
# ax.set_xlim(0.25, len(features) + 0.75)
# ax.set_xlabel('Sample name')
fig= plt.gcf()
fig.set_size_inches(18.5*0.6, 10.5*0.6)



#%%

# Violin plot of all data
plot_data=[results_df['box_precision'].astype(float),results_df['class_precision'].astype(float),results_df['box_recall'].astype(float),results_df['class_recall'].astype(float),results_df['box_f1'].astype(float),results_df['class_f1'].astype(float)]

#%%


data
#%%#############################################
# Select best model and evaluate on test data



data=results_df[results_df.avg_f1 >=0.96]

for best_idx in data.index.values:


    model_name='Hyp__'+'opti='+ results_df.loc[best_idx, 'optimiser'] +'_'+'lr_init='+str(results_df.loc[best_idx, 'train_learn_rate'])+'_'+'lr='+results_df.loc[best_idx, 'learn_rate']+'_'+'bS='+str(results_df.loc[best_idx, 'batch_size'])
    model_path = 'DF_models/Hyper_testing/'+model_name+'.pt'


    m = main.deepforest()
    m.model.load_state_dict(torch.load(model_path))

    results= DFeval(m, test_file , validation_iou_threshold,save_dir)

    f1 = 2*((results['box_recall']*results['box_precision'])/(results['box_recall']+results['box_precision']))


    print(model_name, '\n', str(best_idx) , '|  # Seedlings=', results['results'].match.sum(),' | Precision is= ',results['box_precision'],' | Recall is= ', results['box_recall'],' | f1=',f1, '\n')

#%%#############################################
# Plot loss for a specific index

# results_df=results_df[results_df.learn_rate !='reduce_on_plat']
# results_df.replace('default', 'RoP',inplace=True)
# results_df.replace('StepLR', 'Stepped',inplace=True)
# results_df.replace('exponential', 'Exponential',inplace=True)
# results_df
# results_df.sort_values('avg_f1',ascending=False).head(4)
# results_df[results_df.avg_f1 >=0.96].head(5)
# data = results_df[results_df.avg_f1 == results_df.avg_f1.max()]

results_df.sort_values('avg_f1',ascending=False).head(5)
#%%
results_df.reset_index(drop=True, inplace=True)
data = results_df.sort_values('avg_f1',ascending=False).head(5)
loss_df=pd.DataFrame()
#%%

# epoch2
#%%
# sns.color_palette()
# label=['Optimiser=SGD lr=Stepped Batch Size=1','Optimiser=SGD lr=Stepped Batch Size=2','Optimiser=SGD lr=Stepped Batch Size=3',                      'Optimiser=SGD lr=RoP       Batch Size=4']
# lbl_size=22
# tik_size=lbl_size-4

for idx, i in enumerate(data.index.values):
    best_idx=i

    # version=results_df.loc[best_idx, 'batch_size']-1
    folder = 'Hyp__'+'opti='+ results_df.loc[best_idx, 'optimiser'] +'_'+'lr_init='+str(results_df.loc[best_idx, 'train_learn_rate'])+'_'+'lr='+results_df.loc[best_idx, 'learn_rate']+'_'+'bS='+str(results_df.loc[best_idx, 'batch_size'])
    res=pd.read_csv(r'logs\%s\version_0\metrics.csv' %(folder))
    res.groupby('epoch')
    res1=res.iloc[:,0:4].dropna(axis=0)
    res2=res.iloc[:,2:6].dropna(axis=0)
    lossH=res1.merge(res2,left_on=['epoch','step'],right_on=['epoch','step'],how='inner')
    lossH['val_loss'] = lossH.val_classification + lossH.val_bbox_regression
    lossH['train_loss'] = lossH.train_classification_epoch + lossH.train_bbox_regression_epoch


    # name = 'Optimiser='+ results_df.loc[best_idx, 'optimiser'] +'_'+'lr='+str(results_df.loc[best_idx, 'train_learn_rate'])+'_'+'lr schedule='+results_df.loc[best_idx, 'learn_rate']+'_'+'Batch Size='+str(results_df.loc[best_idx, 'batch_size'])

    name = results_df.loc[best_idx, 'optimiser'] +'_'+str(results_df.loc[best_idx, 'train_learn_rate'])+'_'+results_df.loc[best_idx, 'learn_rate']+'_'+str(results_df.loc[best_idx, 'batch_size'])
    
    lossH['model']=name

    loss_df = loss_df.append(lossH)

    # label= 'Optimiser='+results_df.loc[best_idx, 'optimiser']+' | '+'lr='+results_df.loc[best_idx, 'learn_rate']+' | '+'Batch size='+str(results_df.loc[best_idx, 'batch_size'])
    # plt.figure(figsize=(8, 6), dpi=80)

#%%

loss_df.model.unique()
#%%

# Training and validation loss plot

plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)
lbl_size=30
sns.color_palette()

legend_labels=['_Optimiser=Adam  lr=0.0001  Schedule=RoP         Batch=3',
               'Optimiser=SGD    lr=0.001    Schedule=RoP    Batch=4',
'_Optimiser=Adam  lr=0.0001  Schedule=Stepped   Batch=2',
'_Optimiser=Adam  lr=0.0001  Schedule=RoP          Batch=2',
'_Optimiser=SGD    lr=0.001    Schedule=Stepped    Batch=1']


# eps=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
eps=[1,3,5,7,9,11,13,15]
models=loss_df.model.unique()

palette = {c:'red' if c==models[1] else 'grey' for c in loss_df.model.unique()}

ax = sns.lineplot(x='epoch', y='val_loss', data=loss_df, hue='model', alpha=1, palette=palette, legend=False)#, linewidth = 2)# legend=label)
# ax = sns.lineplot(x='epoch', y='train_loss', data=loss_df, hue='model', alpha=1, palette=palette, legend=False)#, linewidth = 2)# legend=label)

legend1 = ax.legend(legend_labels, loc="upper right", title=None,fontsize=20)
ax.add_artist(legend1)
# ax.set_fontsize('6')

tik_size=lbl_size
# plt.legend(fontsize=tik_size)

# ax.set_ylabel('Validation loss',fontsize=lbl_size,labelpad=10)
ax.set_ylabel('FL - validation data',fontsize=lbl_size,labelpad=10)
# ax.set_ylabel('Training loss',fontsize=lbl_size,labelpad=10)
# ax.set_ylabel('FL - training data',fontsize=lbl_size,labelpad=10)

ax.set_xlabel('Training length [epoch]',fontsize=lbl_size,labelpad=10)
ax.get_xaxis().set_tick_params(labelsize=tik_size)
ax.get_yaxis().set_tick_params(labelsize=tik_size)
ax.set_xticklabels(eps)
ax.set_xlim(0, 14)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

image_name='Hypers_validation_loss_fl.png'
# image_name='Hypers_train_loss_fl.png'

image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200)






# ['Adam_0.0001_default_3', 'sgd_0.001_default_4',else c:'blue' if c==models[0] 

#        'Adam_0.0001_StepLR_2', 'Adam_0.0001_default_2',
#        'sgd_0.001_StepLR_1']

# ,alpha=0.8
# legend_labelst=[None,'asda',None]




# palette = {c:'red' if c=='US' else 'grey' for c in df.country.unique()}

# sns.lineplot(x='year', y='pop', data=df, hue='country', 

#              palette=palette, legend=False)




# plt.plot(lossH.epoch,lossH.val_loss, label=label[idx])
# plt.plot(lossH.epoch,lossH.train_loss, label=label)
# plt.legend(fontsize=tik_size)
# sns.despine(offset=10, trim=True, bottom=True, left=True)

# ax.get_legend.



#%%

# Check what happens at epochs=7
data
#%%
# data=resultsA_df[resultsA_df.avg_f1 >= 0.96]

































#%%#############################################
# Continue with best hyperparameters

train_learning_rate = 0.001
learn_rate = 'default'
optimiser = 'sgd'
batch_size=4
epoch=1

#%%#############################################
#  Evaluate AUGMENTATION
#############################################



# m.config["nms_thresh"]
# m.config["score_thresh"]



train_learning_rate = 0.001
learn_rate = 'default'
optimiser = 'sgd'
batch_size = 4
validation_iou_threshold = 0.4
# Best epoch plus one higher to see if Aug needs more epochs
epochs = [4, 15]
min_vis_list = [0.5,0.8]



tileSize = 1000
tileOverlap = 0.25
Aug_images_required_train = 20
Augmentations = ['HF_RBC']#['all', 'SSR','HF','RBC','RR']


idx = 0
resultsA_df = pd.DataFrame(columns=['Augmentation','min_vis','epochs','aug_images','batch_size', 'learn_rate', 'optimiser', 'box_precision', 'box_recall', 'box_f1', 'class_precision', 'class_recall', 'class_f1', 'miou','train_time','aug_time'])

for min_vis in min_vis_list:
    for Augmentation in Augmentations:

        #############################################
        #  AUGMENT IMAGES AND GET ANNOTATIONS

        clean_augmented()

        # train data
        tr_augmented_annotations = augment(tr_tile_annotations, Aug_images_required_train, Train_dir,Augmentation,min_vis) 
        start_time = time.time()
        # val data
        val_augmented_annotations = augment(val_tile_annotations, round(Aug_images_required_train*(validation_split/train_split)), Val_dir,Augmentation,min_vis) 
        aug_time = time.time() - start_time


        #############################################
        #  SAVE ANNOTATIONS

        # train data
        train_anno_final = tr_augmented_annotations.copy()
        annotations_file = save_annotations(train_anno_final, Train_dir)

        # val data
        val_anno_final = val_augmented_annotations.copy()
        validation_file = save_annotations(val_anno_final, Val_dir)

        # test data
        clean_test()
        test_FileNames = glob.glob(r'Agisoft\Ortho\test\*')
        for name in test_FileNames:
            shutil.copy(name, Test_dir)
        test_file = Test_dir + 'test.csv'



        #############################################
        #  Evaluate on validation data

        
        for epoch in epochs:

            file_name ='Augs__' + 'aug=' + Augmentation +'_'+'minVis='+ str(min_vis) + '_' +'epoch=' + str(epoch)+'___run3_'
            # Configure model
            m = config (annotations_file, validation_file, batch_size, nms_threshold, score_threshold, epoch, train_learning_rate, validation_iou_threshold,optimiser,learn_rate,file_name)
            
            start_time = time.time()
            m.trainer.fit(m)
            train_time = time.time() - start_time

            model_path = 'DF_models/Augmentation_testing/'+file_name+'_v3.pt'
            torch.save(m.model.state_dict(),model_path)
            
            # m.config["score_thresh"] = 0.5
            save_dir = 'Results/df_training/Aug_eval_results'


            # Evaluate model
            results = DFeval(m, validation_file, validation_iou_threshold,save_dir)

            resultsA_df.loc[idx, 'aug_time'] = aug_time
            resultsA_df.loc[idx, 'train_time'] = train_time
            resultsA_df.loc[idx, 'Augmentation'] = Augmentation
            resultsA_df.loc[idx, 'min_vis'] = min_vis
            resultsA_df.loc[idx, 'epochs'] = epoch
            resultsA_df.loc[idx, 'aug_images'] = Aug_images_required_train
            resultsA_df.loc[idx, 'learn_rate'] = learn_rate
            resultsA_df.loc[idx, 'optimiser'] = optimiser
            resultsA_df.loc[idx, 'batch_size'] = batch_size
            resultsA_df.loc[idx, 'box_precision'] = results['box_precision']
            resultsA_df.loc[idx, 'box_recall'] = results['box_recall']
            true_positive = results['results']['match'].sum()
            resultsA_df.loc[idx, 'class_recall'] = true_positive / val_anno_final.shape[0]
            resultsA_df.loc[idx, 'class_precision'] = true_positive / results['results'].shape[0]
            resultsA_df.loc[idx, 'miou'] = results['results']['IoU'].mean()

            if resultsA_df.loc[idx, 'box_recall'] + resultsA_df.loc[idx, 'box_precision'] == 0: resultsA_df.loc[idx, 'box_f1'] = 0
            else: resultsA_df.loc[idx, 'box_f1'] = 2*((resultsA_df.loc[idx, 'box_recall']*resultsA_df.loc[idx, 'box_precision'])/(resultsA_df.loc[idx, 'box_recall']+resultsA_df.loc[idx, 'box_precision']))

            if resultsA_df.loc[idx, 'class_recall'] + resultsA_df.loc[idx, 'class_precision'] == 0: resultsA_df.loc[idx, 'class_f1'] = 0
            else: resultsA_df.loc[idx, 'class_f1'] = 2*((resultsA_df.loc[idx, 'class_recall']*resultsA_df.loc[idx, 'class_precision'])/(resultsA_df.loc[idx, 'class_recall']+resultsA_df.loc[idx, 'class_precision']))

            # Add average metrics
            resultsA_df['avg_precision'] = (resultsA_df['box_precision'] + resultsA_df['class_precision'])*0.5
            resultsA_df['avg_recall'] = (resultsA_df['box_recall'] + resultsA_df['class_recall'])*0.5
            resultsA_df['avg_f1'] = 2*((resultsA_df['avg_recall']*resultsA_df['avg_precision'])/(resultsA_df['avg_recall']+resultsA_df['avg_precision']))

            idx += 1
            # model_path = 'DF_models/Augmentation_testing/'+ file_name + '_v2.pt'
            # torch.save(m.model.state_dict(),model_path)

            # clear cuda memory
            torch.cuda.empty_cache()
            del m
            gc.collect()

resultsA_df.to_csv('Results/DeepForest_results/Augmentation_results_v3_HF_RBC.csv', index=False)

#%%

resultsA_df=pd.read_csv('Results/DeepForest_results/Augmentation_results.csv',delimiter=',')

#%%
# resultsA_df.sort_values('avg_f1', ascending=False)
# resultsA_df=temp.copy()
# temp2 =  resultsA_df.copy()
# temp2
resultsA_df.groupby('min_vis')['train_time'].mean()
#%%
resultsA_df[(resultsA_df.avg_f1 >= 0.96)].sort_values('avg_f1', ascending=False)
#%%


#%%
# resultsA_df[(resultsA_df.epochs==15)&(resultsA_df.avg_f1 >= 0.96)].sort_values('avg_f1', ascending=False)

#%%

# Add average metrics
resultsA_df['avg_precision'] = (resultsA_df['box_precision'] + resultsA_df['class_precision'])*0.5
resultsA_df['avg_recall'] = (resultsA_df['box_recall'] + resultsA_df['class_recall'])*0.5
resultsA_df['avg_f1'] = 2*((resultsA_df['avg_recall']*resultsA_df['avg_precision'])/(resultsA_df['avg_recall']+resultsA_df['avg_precision']))


#%%
# best_idx
# data=resultsA_df[resultsA_df.avg_f1 >= 0.94]
data=resultsA_df[resultsA_df.Augmentation == 'none']
# data.index.values
data
#%%

'Augs__' + 'aug=' + resultsA_df.loc[best_idx, 'Augmentation'] +'_'+'minVis='+ str(resultsA_df.loc[best_idx, 'min_vis']) + '_' +'epoch=' + str(resultsA_df.loc[best_idx, 'epochs'])
# resultsA_df


#%%#############################################
# Evaluate chosen models on test data
# resultsA_df
data

#%%
data=resultsA_df[resultsA_df.avg_f1 >= 0.96]

# data=resultsA_df
# [1,13]
for i in data.index.values:
    
    best_idx=i
    # best_idx = results_df[results_df.box_f1 == results_df.box_f1.max()].index.values[0]

    model_name = 'Augs__' + 'aug=' + resultsA_df.loc[best_idx, 'Augmentation'] +'_'+'minVis='+ str(resultsA_df.loc[best_idx, 'min_vis']) + '_' +'epoch=' + str(resultsA_df.loc[best_idx, 'epochs'])+'___run3_v3'
    model_path = 'Df_models/Augmentation_testing/'+model_name+'.pt'
    m = main.deepforest()
    m.model.load_state_dict(torch.load(model_path))

    results= DFeval(m, test_file, validation_iou_threshold,save_dir)

    f1 = 2*((results['box_recall']*results['box_precision'])/(results['box_recall']+results['box_precision']))
    print(model_name, '\n', str(best_idx) , '|  # Seedlings=', results['results'].match.sum(),' | Precision is= ',results['box_precision'],' | Recall is= ', results['box_recall'],' | f1=',f1, '\n')


#%%#############################################
# Plot loss for a specific index

# data=resultsA_df[resultsA_df.avg_f1 >= 0.96]
# data=resultsA_df[resultsA_df.epochs == 15]
# file_name = 'Augs__' + 'aug=' + resultsA_df.loc[best_idx, 'Augmentation'] +'_'+'minVis='+ str(resultsA_df.loc[best_idx, 'min_vis']) + '_' +'epoch=' + str(resultsA_df.loc[best_idx, 'epochs'])
# model_path = 'Df_models/Augmentation_testing/'+file_name+'.pt'
# data=resultsA_df
for i in data.index.values:
    best_idx=i

    # version=results_df.loc[best_idx, 'batch_size']-1
    folder = 'Augs__' + 'aug=' + resultsA_df.loc[best_idx, 'Augmentation'] +'_'+'minVis='+ str(resultsA_df.loc[best_idx, 'min_vis']) + '_' +'epoch=' + str(resultsA_df.loc[best_idx, 'epochs'])+'___run3'
    res=pd.read_csv(r'logs\%s\version_0\metrics.csv' %(folder))
    res.groupby('epoch')

    res1=res.iloc[:,0:4].dropna(axis=0)
    # res1
    res2=res.iloc[:,2:6].dropna(axis=0)
    # res2
    lossA=res1.merge(res2,left_on=['epoch','step'],right_on=['epoch','step'],how='inner')
    lossA
    lossA['val_loss'] = lossA.val_classification + lossA.val_bbox_regression
    lossA['train_loss'] = lossA.train_classification_epoch + lossA.train_bbox_regression_epoch

    label='index=',i,'  |aug=' + resultsA_df.loc[best_idx, 'Augmentation'] +'_'+'minVis='+ str(resultsA_df.loc[best_idx, 'min_vis']) + '_' +'epoch=' + str(resultsA_df.loc[best_idx, 'epochs'])
    # plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(lossA.epoch,lossA.val_loss, label=label)
    plt.legend()

fig = plt.gcf()
# fig.legend()
fig.set_size_inches(18.5, 10.5)

#%%

#############################################
#  CONTINUE WITH BEST AUGMENTAIONS SETTINGS AND BEST MODEL

model_name='Augs__aug=RBC_minVis=0.3_epoch=4___run3'
model_path = 'Df_models/Augmentation_testing/'+model_name+'.pt'
m = main.deepforest()
m.model.load_state_dict(torch.load(model_path))

batch_size=4
train_learning_rate = 0.001

validation_iou_threshold=0.4

min_vis = 0.3
Augmentation = 'none'
Aug_images_required_train = 0


#############################################
#  AUGMENT IMAGES AND GET ANNOTATIONS

clean_augmented()

# train data
tr_augmented_annotations = augment(tr_tile_annotations, Aug_images_required_train, Train_dir, Augmentation, min_vis) 

# val data
val_augmented_annotations = augment(val_tile_annotations, round(Aug_images_required_train*(validation_split/train_split)), Val_dir, Augmentation, min_vis) 



#############################################
#  SAVE ANNOTATIONS

# train data
train_anno_final = tr_augmented_annotations.copy()
annotations_file = save_annotations(train_anno_final, Train_dir)

# val data
val_anno_final = val_augmented_annotations.copy()
validation_file = save_annotations(val_anno_final, Val_dir)

# test data
clean_test()
test_FileNames = glob.glob(r'Agisoft\Ortho\test\*')
for name in test_FileNames:
    shutil.copy(name, Test_dir)
test_file = Test_dir + 'test.csv'


#%%#############################################
#  EVALUATE PREDICTION SETTINGS -Patch size, overlap, nms_threshold, score_threshold
#############################################

# Load the chosen model
# model_pathF='DF_models/Augmentation_testing/Augs__aug=RBC_minVis=0.3_epoch=3.pt'
# model_pathF='DF_models/Threshold_testing/Thresholds__nms=0.3_score=0.5.pt'
model_pathF='DF_models/Report models/Hyper models/Augs__aug=none_minVis=2_epoch=4___run3.pt'

m = main.deepforest()
m.model.load_state_dict(torch.load(model_pathF))


patch_overlaps=[0.1,0.2,0.25,0.3]
patch_sizes=[900,925,950,975,1000]
nms_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
score_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

# t_anno=pd.DataFrame()

# t_anno=t_anno.append(train_annotations)
# t_anno=t_anno.append(val_annotations)

# anno=[train_annotations, val_annotations, test_annotations]
# names=['train','val','test']
anno=[val_annotations]
names=['val']
resultsP_df = pd.DataFrame(columns=['set','Data','nms_threshold','score_threshold','patch_size','patch_overlap','predicted','missing','found','avg_precision','avg_recall','avg_f1','box_precision', 'box_recall', 'box_f1', 'class_precision', 'class_recall', 'class_f1', 'miou', 'prediction_time'])

idx=0
for scr in score_thresholds:
    for nms in nms_thresholds:
        for patch_overlap in patch_overlaps:
            for patch_size in patch_sizes:
                

                m.config["nms_thresh"] = nms
                m.config["score_thresh"] = scr

                # t_boxes=pd.DataFrame()
                # tr_boxes=pd.DataFrame()
                val_boxes=pd.DataFrame()
                # te_boxes=pd.DataFrame()

                # for i in train_annotations.image_path.unique():
                #     img,box = m.predict_tile(os.path.join('Agisoft/Ortho/train/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
                #     box['image_path'] = i
                #     tr_boxes=tr_boxes.append(box)

                for i in val_annotations.image_path.unique():
                    start_time = time.time()
                    img ,box = m.predict_tile(os.path.join('Agisoft/Ortho/val/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
                    prediction_time = time.time() - start_time
                    box['image_path'] = i
                    val_boxes=val_boxes.append(box)

                # for i in test_annotations.image_path.unique():
                #     img ,box = m.predict_tile(os.path.join('Agisoft/Ortho/test/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
                #     box['image_path'] = i
                #     te_boxes=te_boxes.append(box)

                # t_boxes=t_boxes.append(tr_boxes)
                # t_boxes=t_boxes.append(val_boxes)

                # data=[tr_boxes,val_boxes, te_boxes]
                data=[val_boxes]

                set=int(idx/1)
                for i, d in enumerate(data):

                    results = evaluate.evaluate(predictions=d, ground_df=anno[i],root_dir=annotations_file, iou_threshold=validation_iou_threshold, savedir=None)   

                    true_positive = results['results']['match'].sum()
                    resultsP_df.loc[idx, 'prediction_time'] = prediction_time
                    resultsP_df.loc[idx, 'set'] = int(set)
                    resultsP_df.loc[idx, 'Data'] = names[i]
                    resultsP_df.loc[idx, 'patch_size'] = patch_size
                    resultsP_df.loc[idx, 'nms_threshold'] = nms
                    resultsP_df.loc[idx, 'score_threshold'] = scr
                    resultsP_df.loc[idx, 'predicted'] = d.shape[0]
                    resultsP_df.loc[idx, 'missing'] = anno[i].shape[0]-true_positive
                    resultsP_df.loc[idx, 'found'] = true_positive
                    resultsP_df.loc[idx, 'patch_overlap'] = patch_overlap
                    resultsP_df.loc[idx, 'box_precision'] = results['box_precision']
                    resultsP_df.loc[idx, 'box_recall'] = results['box_recall']
                    
                    resultsP_df.loc[idx, 'class_recall'] = true_positive / anno[i].shape[0]
                    resultsP_df.loc[idx, 'class_precision'] = true_positive / (results['results'].shape[0])
                    resultsP_df.loc[idx, 'miou'] = results['results']['IoU'].mean()

                    if resultsP_df.loc[idx, 'box_recall'] + resultsP_df.loc[idx, 'box_precision'] == 0: resultsP_df.loc[idx, 'box_f1'] = 0
                    else: resultsP_df.loc[idx, 'box_f1'] = 2*((resultsP_df.loc[idx, 'box_recall']*resultsP_df.loc[idx, 'box_precision'])/(resultsP_df.loc[idx, 'box_recall']+resultsP_df.loc[idx, 'box_precision']))

                    if resultsP_df.loc[idx, 'class_recall'] + resultsP_df.loc[idx, 'class_precision'] == 0: resultsP_df.loc[idx, 'class_f1'] = 0
                    else: resultsP_df.loc[idx, 'class_f1'] = 2*((resultsP_df.loc[idx, 'class_recall']*resultsP_df.loc[idx, 'class_precision'])/(resultsP_df.loc[idx, 'class_recall']+resultsP_df.loc[idx, 'class_precision']))
                    

                    resultsP_df.loc[idx, 'avg_precision'] = (resultsP_df.loc[idx, 'box_precision'] + resultsP_df.loc[idx, 'class_precision'])*0.5
                    resultsP_df.loc[idx, 'avg_recall'] = (resultsP_df.loc[idx, 'box_recall'] + resultsP_df.loc[idx, 'class_recall'])*0.5

                    if resultsP_df.loc[idx, 'avg_recall'] + resultsP_df.loc[idx, 'avg_precision'] == 0: resultsP_df.loc[idx, 'class_f1'] = 0
                    else: resultsP_df.loc[idx, 'avg_f1'] = 2*((resultsP_df.loc[idx, 'avg_recall']*resultsP_df.loc[idx, 'avg_precision'])/(resultsP_df.loc[idx, 'avg_recall']+resultsP_df.loc[idx, 'avg_precision']))
                    idx +=1

resultsP_df.to_csv('Results/DeepForest_results/Patch_overlap_results_1.csv', index=False)
resultsP_df

#%%
#############################################
# Investigate results
resultsP_df


#%%

trP=resultsP_df[resultsP_df.Data=='train']
valP=resultsP_df[resultsP_df.Data=='val']
testP=resultsP_df[resultsP_df.Data=='test']

# valP[valP.avg_f1 >=0.99].sort_values('avg_f1',ascending=False)
valP[valP.avg_f1 ==valP.avg_f1.max()].sort_values('avg_recall',ascending=False).head(20)

#%%

i=137
resultsP_df[resultsP_df.set==i]

#%%






# # Add average metrics
# resultsP_df['avg_precision'] = (resultsP_df['box_precision'] + resultsP_df['class_precision'])*0.5
# resultsP_df['avg_recall'] = (resultsP_df['box_recall'] + resultsP_df['class_recall'])*0.5
# resultsP_df['avg_f1'] = 2*((resultsP_df['avg_recall']*resultsP_df['avg_precision'])/(resultsP_df['avg_recall']+resultsP_df['avg_precision']))

# (resultsP_df[resultsP_df.box_f1 >=0.98].groupby('set')['missing'].sum()==1).index.values


#.sort_values( ascending=True).head(10)
# resultsP_df=tempP.copy()

# a=(resultsP_df[resultsP_df.box_f1 >=0.98].groupby('set')['missing'].sum()==1)

# resultsP_df[(resultsP_df.Data =='val')&(resultsP_df.missing)]
# resultsP_df.groupby('set')



# resultsP_df[resultsP_df['set'][a.index.values]]

# all_index=resultsP_df.index.values


# missing1=[]
# for i in a.index.values
#     index = resultsP_df
#     missing1.append(i)

# a.index

# i=22
# resultsP_df[resultsP_df.set==i]
# a
# # resultsP_df
# # resultsP_df[resultsP_df.box_f1 >=0.98].groupby('set')['missing'].sum().sort_values( ascending=True)


# resultsP_df[resultsP_df.avg_f1 >=0.99].groupby('set').last()


# # resultsP_df[resultsP_df.Data=='val']



#%%
######################################################
# LOCAL MAXIMA - TUNING
######################################################

# Load the chosen model
# model_pathF='DF_models/Augmentation_testing/Augs__aug=RBC_minVis=0.3_epoch=3.pt'

model_name='Augs__aug=none_minVis=2_epoch=4___run3'
model_path = 'Df_models/Report models/Hyper models/'+model_name+'.pt'
m = main.deepforest()
m.model.load_state_dict(torch.load(model_path))

m.config["batch_size"] = 4
m.config["validation"]["iou_threshold"]=0.4
m.config["train"]['epochs']=4
m.config["train"]["optimiser"] = 'sgd'
m.config["train"]["lr_schedule"] = 'default'
Augmentation = 'none'
min_vis=0

patch_overlap=0.1 #0.3
patch_size=900 #950
m.config["nms_thresh"] =0.05 #0.35
m.config["score_thresh"] =0.3 #0.35




distances =[10,20,30,40,50]
max_peaks = [98, 500, 2000]
min_height = [30,40,50,60,70,80,90]
offset = 0
use_max = [False,True]

lm_results = pd.DataFrame(columns=['max','min_distance', 'threshold_rel','max_peaks','missing','mae', 'rmse','mape','r2','search_time'])


#############################################
#  PREDICT IMAGES AND GET BOUNDING BOXES


# train data
tr_imgDF =[]
tr_boxes=[]
tr_trays=[]
for i in train_annotations.image_path.unique():
    # imgT = "Agisoft/Ortho/" + i
    imDF, box = m.predict_tile(os.path.join('Agisoft/Ortho/train/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
    base = i.split('_')[0]
    tr_trays.append(base)
    box['Tray'] = base
    tr_imgDF.append(imDF)
    tr_boxes.append(box)

# val data
val_imgDF =[]
val_boxes=[]
val_trays=[]
for i in val_annotations.image_path.unique():
    # imgT = "Agisoft/Ortho/" + i
    imgDF, box = m.predict_tile(os.path.join('Agisoft/Ortho/val/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
    base = i.split('_')[0]
    val_trays.append(base)
    box['Tray'] = base
    val_imgDF.append(imDF)
    val_boxes.append(box)



#############################################
#  GET CORRESPONDING DTMS AND TRAY NAMES

# train data
tr_dtms=[]
for t in tr_trays:
    d = t + '_DTM.tif'
    tr_dtms.append(d)

# val data
val_dtms=[]
for t in val_trays:
    d = t + '_DTM.tif'
    val_dtms.append(d)


#############################################
# LM tuning
#############################################
for idx_um, um in enumerate(use_max):
    for idx_mh, mh in enumerate(min_height):
        for idx_mp, mp in enumerate(max_peaks):
            for idx_dis, dis in enumerate(distances):

                #############################################
                #  FIND LM

                # train data
                tr_lms =[]
                start_time = time.time()
                for dtm in tr_dtms:
                    temp_lm = lm(dtm, dis, mh, offset, mp)
                    tr_lms.append(temp_lm)
                search_time = time.time() - start_time

                # val data
                val_lms =[]
                for dtm in val_dtms:
                    temp_lm = lm(dtm, dis, mh, offset, mp)
                    val_lms.append(temp_lm)


                #############################################
                # ASSIGN LM

                # train data
                temp = []
                for b, d in enumerate(tr_lms):
                    t = assign_lm(tr_boxes[b], d, um)
                    temp.append(t)
                tr_boxes = temp.copy()

                # val data
                temp = []
                for b, d in enumerate(val_lms):
                    t = assign_lm(val_boxes[b], d)
                    temp.append(t)
                val_boxes = temp.copy()

                #############################################
                # CREATE GRIDS

                # train data
                tr_df_grids =[]
                temp=[]
                for idx, i in enumerate(tr_imgDF):
                    temp = segment(i, tr_boxes[idx])
                    tr_df_grids.append(temp)

                # val data
                val_df_grids =[]
                temp=[]
                for idx, i in enumerate(val_imgDF):
                    temp = segment(i,val_boxes[idx])
                    val_df_grids.append(temp)

                #############################################
                # INDEX SEEDLINGS

                # train data
                temp = []
                for b, g in enumerate(tr_df_grids):
                    t = index(tr_boxes[b], g)
                    temp.append(t)
                tr_boxes = temp.copy()
                trb=joinl(tr_boxes)

                # val data
                temp = []
                for b, g in enumerate(val_df_grids):
                    t = index(val_boxes[b], g)
                    temp.append(t)
                val_boxes = temp.copy()
                valb = joinl(val_boxes)
                #############################################
                # ASSIGN TRUE HEIGHTS

                # train data
                tr_dfFinal=[]
                for b, t in enumerate(tr_trays):
                    height_path = 'Heights/' + t +'_height.xlsx'
                    temp = th (tr_boxes[b], height_path)
                    tr_dfFinal.append(temp)

                # val data
                val_dfFinal=[]
                for b, t in enumerate(val_trays):
                    height_path = 'Heights/' + t +'_height.xlsx'
                    t = th (val_boxes[b], height_path)
                    val_dfFinal.append(t)


                #############################################
                #  EXTRACT SEEDLING CROPS

                # train data
                tr_BoxPos=[]
                tr_cropB=[]
                for d, img in enumerate(tr_imagesO):
                    tbox, tcrop = seedling_extract(tr_dfFinal[d], img)
                    tr_BoxPos.append(tbox)
                    tr_cropB.append(tcrop)

                # val data
                val_BoxPos=[]
                val_cropB=[]
                for d, img in enumerate(val_imagesO):
                    tbox, tcrop = seedling_extract(val_dfFinal[d], img)
                    val_BoxPos.append(tbox)
                    val_cropB.append(tcrop)

                #############################################
                # EXTRACT SEEDLING FEATURES

                # train data
                tr_seedlings=[]
                for idx, c in enumerate(tr_cropB):
                    t = seedling_features(tr_BoxPos[idx], tr_dfFinal[idx], c)
                    tr_seedlings.append(t)

                # val data
                val_seedlings=[]
                for idx, c in enumerate(val_cropB):
                    t = seedling_features(val_BoxPos[idx], val_dfFinal[idx], c)
                    val_seedlings.append(t)

                
                
                tr_seedlings=joinl(tr_seedlings)
                val_seedlings=joinl(val_seedlings)
                tr_seedlings=tr_seedlings.append(val_seedlings)

                lm_index = idx_dis + idx_mp*len(distances) + idx_mh*len(distances)*len(max_peaks)+ idx_um*len(distances)*len(max_peaks)*len(min_height)

                lm_results.loc[lm_index,'search_time'] = search_time
                lm_results.loc[lm_index,'max'] = um
                lm_results.loc[lm_index,'min_height'] = mh
                lm_results.loc[lm_index,'min_distance'] = dis
                lm_results.loc[lm_index,'max_peaks'] = mp
                lm_results.loc[lm_index,'missing'] = trb[pd.isna(trb.Height) == True].shape[0]#tr_seedlings[pd.isna(tr_seedlings.SampledHeight) == True].shape[0]
                tr_seedlings_valid = tr_seedlings #.dropna(axis=0)
                lm_results.loc[lm_index,'rmse'] = np.sqrt(mean_squared_error(tr_seedlings_valid.true_height,tr_seedlings_valid.SampledHeight))
                lm_results.loc[lm_index,'r2'] = r2_score(tr_seedlings_valid.true_height,tr_seedlings_valid.SampledHeight)
                lm_results.loc[lm_index,'mae'] = mean_absolute_error(tr_seedlings_valid.true_height,tr_seedlings_valid.SampledHeight)
                lm_results.loc[lm_index,'mape'] = mean_absolute_percentage_error(tr_seedlings_valid.true_height,tr_seedlings_valid.SampledHeight)
                # INCLUDE TIME TO FIND MINIMA AS A PERFORMANCE MEASURE

                print('\n************************\nIteration=', lm_index,'\nmax_peaks=', mp ,'\nDistance=', dis,"\n************************")

# lm_results.to_csv('Results/LM_results/LM_tuning_results_v3.csv', index=False)
# lm_results

#%%#############################################
#  Investigate results


lm_results=pd.read_csv('Results/LM_results/LM_tuning_results_v3.csv', delimiter=',')

#%%



lm_results_train=lm_results.copy()
lm_results_train
#%%
lm_results
# tr_boxes
#%%
# countNB = val_boxes[0][pd.isna(tr_boxes.Height) == True].shape[0]
# countNB
trb[pd.isna(trb.Height) == True].shape[0]

#%%
tb=joinl(tr_boxes)
tb
#%%
# lm_results[lm_results.r2 == lm_results.r2.max()]
# lm_results[lm_results.r2 == lm_results.r2.max()]
# lm_results[lm_results.mae == lm_results.mae.min()]

# lm_results[lm_results.mape == lm_results.mape.min()]
# lm_results[lm_results.rmse == lm_results.rmse.min()]
# lm_results[lm_results.mae <=27.5]
# lm_results[lm_results.mape <=0.165].sort_values('r2', ascending=False)
# lm_results[lm_results.min_distance == 20].sort_values('r2', ascending=False)



# lm_results[lm_results.missing == 0].sort_values('r2', ascending=False)

# lm_results[lm_results.missing == 0].sort_values('mae', ascending=True).head(5)



test=lm_results.groupby('min_distance')['mae'].values
test


#%%
test.values
#%%

lm_results.min_distance.unique()


#%%

lm_results.astype(float)

#%%

#########################################
# PLOT min_dist - HORIZONTAL

# labels=['0.001', '0.01', '0.1']
lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)



# # violin plot
# ax = sns.violinplot(x=lm_results.mae.astype(float), y=lm_results.max_peaks.astype(float),orient='h')#, palette=my_pal)

# for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
#     violin.set_alpha(alpha)

xaxis=[98,500,2000]
ax=sns.barplot(x=lm_results.min_distance.unique(), y=lm_results.groupby('min_distance')['mae'].mean().values)# data=lm_results.astype(float))



# ax.set_xlim(0, 1.002)
# plt.xlim(plt.xlim()[0], 1.0)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

# ax.set(ylabel=None)
# ax.set_xlabel('Average F1 score',fontsize=lbl_size, labelpad=10)


ax.set_ylabel('MAE [mm]',fontsize=lbl_size, labelpad=10)
ax.set_xlabel('Minimum distance [mm]',fontsize=lbl_size, labelpad=10)

# ax.set_yticklabels(labels)
# ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
# fig.set_size_inches(8,4)

# image_name='Hypers_Initial_lr.png'
image_name='LM_min_dist.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%

#########################################
# PLOT STRATEGY - HORIZONTAL

labels=['Distance to cetroid','Use max']
lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)



# # violin plot
# ax = sns.violinplot(x=lm_results.mae.astype(float), y=lm_results.max_peaks.astype(float),orient='h')#, palette=my_pal)

# for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
#     violin.set_alpha(alpha)



xaxis=[98,500,2000]
ax=sns.barplot(x=lm_results['max'].unique(), y=lm_results.groupby('max')['mae'].mean().values)# data=lm_results.astype(float))



# ax.set_xlim(0, 1.002)
# plt.xlim(plt.xlim()[0], 1.0)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

# ax.set(ylabel=None)
# ax.set_xlabel('Average F1 score',fontsize=lbl_size, labelpad=10)


ax.set_ylabel('MAE [mm]',fontsize=lbl_size, labelpad=10)
ax.set_xlabel('Assignment strategy',fontsize=lbl_size, labelpad=10)

ax.set_xticklabels(labels)
# ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
# fig.set_size_inches(3,4)

# image_name='Hypers_Initial_lr.png'
image_name='LM_strategy.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%#########################################
# PLOT max peaks - HORIZONTAL

labels=['Use max', 'Distance to cetroid']
lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)



# # violin plot
# ax = sns.violinplot(x=lm_results.mae.astype(float), y=lm_results.max_peaks.astype(float),orient='h')#, palette=my_pal)

# for violin, alpha in zip(ax.collections[::2], [0.8,0.8,0.8,0.8]):
#     violin.set_alpha(alpha)



# xaxis=[98,500,2000]
ax=sns.barplot(x=lm_results['max_peaks'].unique(), y=lm_results.groupby('max_peaks')['mae'].mean().values)# data=lm_results.astype(float))



# ax.set_xlim(0, 1.002)
# plt.xlim(plt.xlim()[0], 1.0)
sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="ticks")
sns.set_theme(style="whitegrid")
ax.tick_params(left=False)

# ax.set(ylabel=None)
# ax.set_xlabel('Average F1 score',fontsize=lbl_size, labelpad=10)


ax.set_ylabel('MAE [mm]',fontsize=lbl_size, labelpad=10)
ax.set_xlabel('Maximum allowed peaks',fontsize=lbl_size, labelpad=10)

# ax.set_xticklabels(labels)
# ticklbls=ax.get_yticklabels(which='both')
# for x in ticklbls:
#     x.set_ha('left')


ax.get_yaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=0)
ax.get_xaxis().set_tick_params(labelsize=lbl_size)
plt.tight_layout()
fig = plt.gcf()
# fig.set_size_inches(3,4)

# image_name='Hypers_Initial_lr.png'
image_name='LM_max_p.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')


#%%
#########################################
# PLOT LM all

# results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# results_df_valid=results_df.dropna(axis=0)

# ax[0, 0].plot(range(10), 'r') #row=0, col=0
# ax[1, 0].plot(range(10), 'b') #row=1, col=0
# ax[0, 1].plot(range(10), 'g') #row=0, col=1
# ax[1, 1].plot(range(10), 'k') #row=1, col=1


labels_strat=['Distance to cetroid','Use max']
plt.rcParams["font.family"] = "Times New Roman"
lbl_size=17

fig, ax =plt.subplots(2,2, sharey=True, gridspec_kw = {'wspace':0.1, 'hspace':0.2})

# fig.subplots_adjust(wspace=0, hspace=10)
sns.set_theme(style="whitegrid")
import matplotlib.gridspec as gridspec

# gs1 = gridspec.GridSpec(4,4)
# gs1.update(wspace=0.1025, hspace=0.05) # set the spacing between axes.

sns.barplot(x=lm_results['max'].unique(), y=lm_results.groupby('max')['mae'].mean().values,ax=ax[0,0])
ax[0,0].set_xlabel('Assignment strategy',fontsize=lbl_size, labelpad=10)
ax[0,0].set_xticklabels(labels_strat)
ax[0,0].set_ylabel('mean MAE [mm]',fontsize=lbl_size, labelpad=10)
ax[0,0].get_xaxis().set_tick_params(labelsize=lbl_size)
ax[0,0].get_yaxis().set_tick_params(labelsize=lbl_size)


sns.barplot(x=lm_results['min_distance'].unique(), y=lm_results.groupby('min_distance')['mae'].mean().values,ax=ax[0,1])
ax[0,1].set_xlabel('Minimum distance [mm]',fontsize=lbl_size, labelpad=10)
ax[0,1].get_xaxis().set_tick_params(labelsize=lbl_size)

sns.barplot(x=lm_results['max_peaks'].unique(), y=lm_results.groupby('max_peaks')['mae'].mean().values,ax=ax[1,0])
ax[1,0].set_xlabel('Maximum allowed peaks',fontsize=lbl_size, labelpad=10)
# ax[2].set_xticklabels(labels_strat)
ax[1,0].set_ylabel('mean MAE [mm]',fontsize=lbl_size, labelpad=10)
ax[1,0].get_xaxis().set_tick_params(labelsize=lbl_size)
ax[1,0].get_yaxis().set_tick_params(labelsize=lbl_size)


sns.barplot(x=lm_results['min_height'].unique().astype(int), y=lm_results.groupby('min_height')['mae'].mean().values,ax=ax[1,1])
ax[1,1].set_xlabel('Maximum allowed height [mm]',fontsize=lbl_size, labelpad=10)
ax[1,1].get_xaxis().set_tick_params(labelsize=lbl_size)

# for a in ax:
#     a.set_xticklabels([])
#     a.set_yticklabels([])
#     a.set_aspect('equal')
# fig.subplots_adjust(wspace=0.1,hspace=0.1)

sns.despine(offset=10, bottom=True, left=True)

# plt.ylabel('Average F1 score',fontsize=lbl_size)
# ax[0][1].plot
# ax[0].set_ylabel('MAE [mm]',fontsize=lbl_size, labelpad=10)
# ax[0].set_xlabel('Optimiser',fontsize=lbl_size)
# ax[0].set_ylim(0, 1)
# ax[0].get_xaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=10)
# ax[0].get_yaxis().set_tick_params(labelsize=lbl_size)


# ax[1].plot
# ax[1].set_ylabel('MAE [mm]',fontsize=lbl_size, labelpad=10)
# ax[1].set_xticklabels(labels_strat)
# sns.violinplot(y=results_df_valid.avg_f1,x=results_df_valid.learn_rate, data=results_df_valid, ax=ax[1])
# ax[1].set_ylabel(None)
# ax[1].get_xaxis().set_tick_params(labelsize=lbl_size, direction='out',pad=10)
# ax[1].get_yaxis().set_tick_params(labelsize=lbl_size)




# 

# ax.tick_params(left=False)
# ax.set_xticklabels(labels)



# plt.xlabel('Optimiser',fontsize=lbl_size)
plt.tight_layout()
# fig = plt.gcf()
fig.set_size_inches(12,12)





image_name='LM mean MAE all.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')


plt.show()



#%%#############################################
# Plot performance measures

fig, axs = plt.subplots(3, sharex=True)
fig.set_size_inches(18.5, 18.5)

for mp in max_peaks:
    axs[0].plot(lm_results[lm_results.max_peaks==mp].min_distance, lm_results[lm_results.max_peaks==mp].RMSE)
    axs[1].plot(lm_results[lm_results.max_peaks==mp].min_distance, lm_results[lm_results.max_peaks==mp].r2)
    axs[2].plot(lm_results[lm_results.max_peaks==mp].min_distance, lm_results[lm_results.max_peaks==mp].MAE)
fig.legend(max_peaks, loc='upper right')
axs[0].set_title('RMSE')
axs[1].set_title('r2')
axs[2].set_title('MAE')

#%%

fig, axs = plt.subplots(3, sharex=True)
fig.set_size_inches(18.5, 18.5)
min_height=[30]
for mh in min_height:
    axs[0].plot(lm_results[lm_results.min_height==mh].min_distance, lm_results[lm_results.min_height==mh].RMSE)
    axs[1].plot(lm_results[lm_results.min_height==mh].min_distance, lm_results[lm_results.min_height==mh].r2)
    axs[2].plot(lm_results[lm_results.min_height==mh].min_distance, lm_results[lm_results.min_height==mh].MAE)
fig.legend()
axs[0].set_title('RMSE')
axs[1].set_title('r2')
axs[2].set_title('MAE')


#%%######################################################
# LOCAL MAXIMA - ON BEST PARAMETERS
######################################################


# LOAD MODEL
model_name='Augs__aug=none_minVis=2_epoch=4___run3'
model_path = 'Df_models/Report models/Hyper models/'+model_name+'.pt'
m = main.deepforest()
m.model.load_state_dict(torch.load(model_path))

m.config["batch_size"] = 4
m.config["validation"]["iou_threshold"]=0.4
m.config["train"]['epochs']=4
m.config["train"]["optimiser"] = 'sgd'
m.config["train"]["lr_schedule"] = 'default'
Augmentation = 'none'
min_vis=0

patch_overlap=0.1 #0.3
patch_size=900 #950
m.config["nms_thresh"] =0.05 #0.35
m.config["score_thresh"] =0.3 #0.35






# CONTINUE WITH LM

m.config["nms_thresh"] =0.05 #0.35
m.config["score_thresh"] =0.3 #0.35
patch_overlap=0.1 #0.3
patch_size=900 #950

# threshold_rel = 
# lm_results = pd.DataFrame(columns=['min_distance', 'threshold_rel','max_peaks','missing', 'RMSE','r2','MAE'])

best_distance = 30
min_height = 40
best_offset = 0
best_max_peaks =2000
use_max = False
#############################################
#  PREDICT IMAGES AND GET BOUNDING BOXES

# train data
tr_imgDF =[]
tr_boxes=[]
tr_trays=[]
for i in train_annotations.image_path.unique():
    # imgT = "Agisoft/Ortho/" + i
    imDF, box = m.predict_tile(os.path.join('Agisoft/Ortho/train/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
    base = i.split('_')[0]
    tr_trays.append(base)
    box['Tray'] = base
    tr_imgDF.append(imDF)
    tr_boxes.append(box)

# val data
val_imgDF =[]
val_boxes=[]
val_trays=[]
for i in val_annotations.image_path.unique():
    # imgT = "Agisoft/Ortho/" + i
    imDF, box = m.predict_tile(os.path.join('Agisoft/Ortho/val/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
    base = i.split('_')[0]
    val_trays.append(base)
    box['Tray'] = base
    val_imgDF.append(imDF)
    val_boxes.append(box)

te_imgDF =[]
te_boxes=[]
te_trays=[]
for i in test_annotations.image_path.unique():
    # imgT = "Agisoft/Ortho/" + i
    imDF, box = m.predict_tile(os.path.join('Agisoft/Ortho/test/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
    base = i.split('_')[0]
    te_trays.append(base)
    box['Tray'] = base
    te_imgDF.append(imDF)
    te_boxes.append(box)


#############################################
#  GET CORRESPONDING DTMS AND TRAY NAMES

# train data
tr_dtms=[]
for t in tr_trays:
    d = t + '_DTM.tif'
    tr_dtms.append(d)

# val data
val_dtms=[]
for t in val_trays:
    d = t + '_DTM.tif'
    val_dtms.append(d)

# test data
te_dtms=[]
for t in te_trays:
    d = t + '_DTM.tif'
    te_dtms.append(d)



#############################################
#  FIND LM

# train data
tr_lms =[]
start_time = time.time()
for dtm in tr_dtms:
    temp_lm = lm(dtm, best_distance, min_height, best_offset, best_max_peaks)
    tr_lms.append(temp_lm)
search_time_train = time.time() - start_time

# val data
val_lms =[]
start_time = time.time()
for dtm in val_dtms:
    temp_lm = lm(dtm, best_distance, min_height, best_offset, best_max_peaks)
    val_lms.append(temp_lm)
search_time_val = time.time() - start_time

# test data
te_lms =[]
start_time = time.time()
for dtm in te_dtms:
    temp_lm = lm(dtm, best_distance, min_height, best_offset, best_max_peaks)
    te_lms.append(temp_lm)
search_time_test = time.time() - start_time

#############################################
# ASSIGN LM

# train data
temp = []
for b, d in enumerate(tr_lms):
    t = assign_lm(tr_boxes[b], d, use_max)
    temp.append(t)
tr_boxes = temp.copy()

# val data
temp = []
for b, d in enumerate(val_lms):
    t = assign_lm(val_boxes[b], d, use_max)
    temp.append(t)
val_boxes = temp.copy()

# test data
temp = []
for b, d in enumerate(te_lms):
    t = assign_lm(te_boxes[b], d, use_max)
    temp.append(t)
te_boxes = temp.copy()


#############################################
# CREATE GRIDS

# train data
tr_df_grids =[] 
temp=[]
for idx, i in enumerate(tr_imgDF):
    temp = segment(i, tr_boxes[idx])
    tr_df_grids.append(temp)

# val data
val_df_grids =[]
temp=[]
for idx, i in enumerate(val_imgDF):
    temp = segment(i,val_boxes[idx])
    val_df_grids.append(temp)

# test data
te_df_grids =[]
temp=[]
for idx, i in enumerate(te_imgDF):
    temp = segment(i, te_boxes[idx])
    te_df_grids.append(temp)

#############################################
# INDEX SEEDLINGS

# train data
temp = []
for b, g in enumerate(tr_df_grids):
    t = index(tr_boxes[b], g)
    temp.append(t)
tr_boxes = temp.copy()

# val data
temp = []
for b, g in enumerate(val_df_grids):
    t = index(val_boxes[b], g)
    temp.append(t)
val_boxes = temp.copy()

# test data
temp = []
for b, g in enumerate(te_df_grids):
    t = index(te_boxes[b], g)
    temp.append(t)
te_boxes = temp.copy()


#############################################
# ASSIGN TRUE HEIGHTS

# train data
tr_dfFinal=[]
for b, t in enumerate(tr_trays):
    height_path = 'Heights/' + t +'_height.xlsx'
    temp = th (tr_boxes[b], height_path)
    tr_dfFinal.append(temp)

# val data
val_dfFinal=[]
for b, t in enumerate(val_trays):
    height_path = 'Heights/' + t +'_height.xlsx'
    temp = th (val_boxes[b], height_path)
    val_dfFinal.append(temp)

# test data
te_dfFinal=[]
for b, t in enumerate(te_trays):
    height_path = 'Heights/' + t +'_height.xlsx'
    temp = th (te_boxes[b], height_path)
    te_dfFinal.append(temp)


#############################################
#  EXTRACT SEEDLING CROPS

# train data
tr_BoxPos=[]
tr_cropB=[]
for d, img in enumerate(tr_imagesO):
    tbox, tcrop = seedling_extract(tr_dfFinal[d], img, tuning=False)
    tr_BoxPos.append(tbox)
    tr_cropB.append(tcrop)

# val data
val_BoxPos=[]
val_cropB=[]
for d, img in enumerate(val_imagesO):
    tbox, tcrop = seedling_extract(val_dfFinal[d], img, tuning=False)
    val_BoxPos.append(tbox)
    val_cropB.append(tcrop)

# test data
# import test images
te_imagesO = []
for image in test_annotations.image_path.unique():
    rasterO = "Agisoft/Ortho/" + image
    img=rasterio.open(rasterO).read()
    img=np.moveaxis(img, 0, 2)
    te_imagesO.append(img)

te_BoxPos=[]
te_cropB=[]
for d, img in enumerate(te_imagesO):
    tbox, tcrop = seedling_extract(te_dfFinal[d], img, tuning=False)
    te_BoxPos.append(tbox)
    te_cropB.append(tcrop)


#############################################
# EXTRACT SEEDLING FEATURES

# train data
tr_seedlings=[]
for idx, c in enumerate(tr_cropB):
    t = seedling_features(tr_BoxPos[idx], tr_dfFinal[idx], c)
    tr_seedlings.append(t)

# val data
val_seedlings=[]
for idx, c in enumerate(val_cropB):
    t = seedling_features(val_BoxPos[idx], val_dfFinal[idx], c)
    val_seedlings.append(t)

# val data
te_seedlings=[]
for idx, c in enumerate(te_cropB):
    t = seedling_features(te_BoxPos[idx], te_dfFinal[idx], c)
    te_seedlings.append(t)




#############################################
# PLOT RESULTS

fig, axs = plt.subplots(4, sharex=True)
fig.set_size_inches(18.5, 18.5)

# for mp in max_peaks:
axs[0].plot(tr_seedlings[0].position, tr_seedlings[0].SampledHeight)
axs[0].plot(tr_seedlings[0].position, tr_seedlings[0].true_height)
axs[1].plot(tr_seedlings[1].position, tr_seedlings[1].SampledHeight)
axs[1].plot(tr_seedlings[1].position, tr_seedlings[1].true_height)
axs[2].plot(val_seedlings[0].position, val_seedlings[0].SampledHeight)
axs[2].plot(val_seedlings[0].position, val_seedlings[0].true_height)
axs[3].plot(te_seedlings[0].position, te_seedlings[0].SampledHeight)
axs[3].plot(te_seedlings[0].position, te_seedlings[0].true_height)
# fig.legend()
axs[0].set_title('Training 1')
axs[1].set_title('Training 2')
axs[2].set_title('Validation')
axs[3].set_title('Test')


tr_seedlings=joinl(tr_seedlings)
val_seedlings=joinl(val_seedlings)
te_seedlings=joinl(te_seedlings)

tr_seedlings_valid=tr_seedlings.dropna(axis=0)
val_seedlings_valid=val_seedlings.dropna(axis=0)
te_seedlings_valid=te_seedlings.dropna(axis=0)

lm_results_final = pd.DataFrame(columns=['Data','RMSE', 'MAE','MAPE','R^2'])
D = ['Train','Validation','Test']
# output = [tr_seedlings_valid,val_seedlings_valid, te_seedlings_valid]
output = [tr_seedlings,val_seedlings, te_seedlings]
times=[search_time_train, search_time_val, search_time_test]


for idx, o in enumerate(output):
    lm_results_final.loc[idx,'Data'] = D[idx]
    lm_results_final.loc[idx,'RMSE'] = np.sqrt(mean_squared_error(o.true_height,o.SampledHeight))
    lm_results_final.loc[idx,'MAE'] = mean_absolute_error(o.true_height,o.SampledHeight)
    lm_results_final.loc[idx,'MAPE'] = mean_absolute_percentage_error(o.true_height,o.SampledHeight)
    lm_results_final.loc[idx,'R^2'] = r2_score(o.true_height,o.SampledHeight)
    lm_results_final.loc[idx,'Time'] = times[idx]
    

lm_results_final.to_csv('Results/LM_results/LM_final_results.csv', index=False)
lm_results_final


#%%
# FEATURE INVESTIGATION

lm_results_final=pd.read_csv('Results/DeepForest_results/Hyperparameter_tuning_results.csv',delimiter=',')






tr_seedlings['errorH_abs']=tr_seedlings['errorH'].abs()
#%%

##################################################################
# CORRELATION OF WORST

worst=tr_seedlings[tr_seedlings.tray=='Tray5'].sort_values('errorH_abs', ascending=False).head(5)

data=worst.drop(columns=['tray'])

lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)

labels_corr=['Position', 'boxWidth', 'boxHeight', 'boxArea', 'meanGreen', 'meanBlue',
       'meanRed', 'SampledHeight', 'neighbour_meanHeight',
       'neighbour_maxHeight', 'edge', 'neighbour_meanHeight_Diff',
       'neighbour_maxHeight_Diff', 'true_height', 'errorH', 'errorH_abs']
ax=sns.heatmap(data.corr(),vmin=-1,vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_yticklabels(
    labels_corr,
    
    horizontalalignment='right'
)
# rotation=45,
ax.set_xticklabels(
    labels_corr,
    rotation=45,
    horizontalalignment='right'
)
# 
image_name='Corr_worst.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')

#%%

##################################################################
# CORRELATION OF Best

best=tr_seedlings[tr_seedlings.tray=='Tray5'].sort_values('errorH_abs', ascending=False).tail(5)



data=best.drop(columns=['tray'])

lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)

labels_corr=['Position', 'boxWidth', 'boxHeight', 'boxArea', 'meanGreen', 'meanBlue',
       'meanRed', 'SampledHeight', 'neighbour_meanHeight',
       'neighbour_maxHeight', 'edge', 'neighbour_meanHeight_Diff',
       'neighbour_maxHeight_Diff', 'true_height', 'errorH', 'errorH_abs']
ax=sns.heatmap(data.corr(),vmin=-1,vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_yticklabels(
    labels_corr,
    
    horizontalalignment='right'
)
# rotation=45,
ax.set_xticklabels(
    labels_corr,
    rotation=45,
    horizontalalignment='right'
)
# 
image_name='Corr_best.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
# plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')


#%%

##################################################################
# CORRELATION OF ALL

data=tr_seedlings.drop(columns=['tray','errorH'])

lbl_size=15
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(dpi=1200)

labels_corr=['Position', 'boxWidth', 'boxHeight', 'boxArea', 'meanGreen', 'meanBlue',
       'meanRed', 'SampledHeight', 'neighbour_meanHeight',
       'neighbour_maxHeight', 'edge', 'neighbour_meanHeight_Diff',
       'neighbour_maxHeight_Diff', 'true_height']
ax=sns.heatmap(data.corr(),vmin=-1,vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_yticklabels(
    labels_corr,
    
    horizontalalignment='right'
)
# rotation=45,
ax.set_xticklabels(
    labels_corr,
    rotation=45,
    horizontalalignment='right'
)
# 
# sns.despine(offset=10, trim=True, left=True, bottom=True)
# sns.set_theme(style="whitegrid")
# ax.tick_params(left=False)


image_name='Corr_all.png'
image_path= r'C:\Users\camer\OneDrive - Stellenbosch University\Documents\MEng\Research Project 876\Report\LaTeX\Images'
plt.savefig(os.path.join(image_path, image_name),dpi=1200,bbox_inches='tight')



# seedlings['neighbour_meanHeight_Diff'] = seedlings['neighbour_meanHeight'] - seedlings['SampledHeight']
# seedlings['neighbour_maxHeight_Diff'] = seedlings['neighbour_maxHeight'] - seedlings['SampledHeight']



#%%
# Side by side plot of best and worst


best=tr_seedlings[tr_seedlings.tray=='Tray5'].sort_values('errorH_abs', ascending=False).tail(5).drop(columns=['tray'])
worst=tr_seedlings[tr_seedlings.tray=='Tray5'].sort_values('errorH_abs', ascending=False).head(5).drop(columns=['tray'])

labels_corr=['Position', 'boxWidth', 'boxHeight', 'boxArea', 'meanGreen', 'meanBlue',
       'meanRed', 'SampledHeight', 'neighbour_meanHeight',
       'neighbour_maxHeight', 'edge', 'neighbour_meanHeight_Diff',
       'neighbour_maxHeight_Diff', 'true_height', 'errorH', 'errorH_abs']
fig, ax =plt.subplots(1,2, sharey=True)


sns.set_theme(style="whitegrid")

sns.heatmap(best.corr(),vmin=-1,vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True, ax=ax[0])
ax[0].set_xlabel('Top five',fontsize=lbl_size, labelpad=10)
ax[0].set_yticklabels(
    labels_corr,
    horizontalalignment='right'
)
ax[0].cbar
# rotation=45,
# ax[0].set_xticklabels(
#     labels_corr,
#     rotation=45,
#     horizontalalignment='right'
# )

sns.heatmap(worst.corr(),vmin=-1,vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True, ax=ax[1])
ax[1].set_xlabel('Bottom five',fontsize=lbl_size, labelpad=10)
ax[1].set_yticklabels(
    labels_corr,
    horizontalalignment='right'
)
# rotation=45,
# ax[1].set_xticklabels(
#     labels_corr,
#     rotation=45,
#     horizontalalignment='right'
# )




#%%



#%%
##########################################################################
# PLOT TRAY
##########################################################################


# worst=tr_seedlings.sort_values('errorH', ascending=True).head(1)
# worst_n=neighbours(worst.position.values[0])
# worst_n


data=tr_seedlings[tr_seedlings.tray=='Tray5']


tray5E=pd.DataFrame(np.zeros([7, 14])*np.nan,columns=['col0','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13'])
for p in data.position:
    r=np.floor(p/14).astype(int)
    c= (p%14)
    # print(r,'    ',c)

    # row.append()
    tray5E['col%d' %c].values[r]=data[data.position==p].errorH.values[0]
    # tray5.iloc[c,r]=data[data.position==p].errorH[0]

tray5H=pd.DataFrame(np.zeros([7, 14])*np.nan,columns=['col0','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13'])
for p in data.position:
    r=np.floor(p/14).astype(int)
    c= (p%14)
    # print(r,'    ',c)

    # row.append()
    tray5H['col%d' %c].values[r]=data[data.position==p].true_height.values[0]
    # tray5.iloc[c,r]=data[data.position==p].errorH[0]






#%%


fig, ax = plt.subplots(figsize=(15,6))  
sns.heatmap(tray5H,cmap='Greens',annot= tray5E.values,linewidths=.5,fmt='')

#%%
fig, ax = plt.subplots(figsize=(15,6))  
sns.heatmap(tray5H,cmap='Greens',annot= tray5E.values,linewidths=.5,fmt='')

#%%
tray5E

#%%#############################################
# PLOT RESULTS

fig, axs = plt.subplots(4, sharex=True)
fig.set_size_inches(18.5, 18.5)

axs[0].bar(tr_seedlings[tr_seedlings.tray=='Tray5'].position-0.2, tr_seedlings[tr_seedlings.tray=='Tray5'].SampledHeight)
axs[0].bar(tr_seedlings[tr_seedlings.tray=='Tray5'].position+0.2, tr_seedlings[tr_seedlings.tray=='Tray5'].true_height)

#%%


fig, axs = plt.subplots(4, sharex=True)
fig.set_size_inches(18.5, 18.5)

# for mp in max_peaks:
axs[0].plot(tr_seedlings[0].position, tr_seedlings[0].SampledHeight)
axs[0].plot(tr_seedlings[0].position, tr_seedlings[0].true_height)
axs[1].plot(tr_seedlings[1].position, tr_seedlings[1].SampledHeight)
axs[1].plot(tr_seedlings[1].position, tr_seedlings[1].true_height)
axs[2].plot(val_seedlings[0].position, val_seedlings[0].SampledHeight)
axs[2].plot(val_seedlings[0].position, val_seedlings[0].true_height)
axs[3].plot(te_seedlings[0].position, te_seedlings[0].SampledHeight)
axs[3].plot(te_seedlings[0].position, te_seedlings[0].true_height)
# fig.legend()
axs[0].set_title('Training 1')
axs[1].set_title('Training 2')
axs[2].set_title('Validation')
axs[3].set_title('Test')

#%%#############################################
# Plot boxes and local maxima


# Plot trays

# boxes = tr_boxes[0].copy()
# df_grid = tr_df_grids[0].copy()
# plt.imshow(tr_imgDF[0][:,:,::-1])

boxes = tr_boxes[1].copy()
df_grid = tr_df_grids[1].copy()
plt.imshow(tr_imgDF[1][:,:,::-1])

# boxes = te_boxes[0].copy()
# df_grid = te_df_grids[0].copy()
# plt.imshow(te_imgDF[0][:,:,::-1])

# boxes = val_boxes[0].copy()
# df_grid = val_df_grids[0].copy()
# plt.imshow(val_imgDF[0][:,:,::-1])


#  Plot local maxima
plt.scatter(boxes[pd.isna(boxes.Height) == False].Lmax_X,boxes[pd.isna(boxes.Height) == False].Lmax_Y,c='r')
# Plot numbers
for i in boxes.index:
    plt.text(boxes.xc[i],boxes.yc[i]+25,boxes.position[i],bbox=dict(boxstyle="round",facecolor='blue', alpha=0.4))
# Plot grid centroids
plt.scatter(df_grid.x_coord, df_grid.y_coord)
fig= plt.gcf()
fig.set_size_inches(18.5*3, 10.5*3)



#%%#######################################################################
# MODELLING
#######################################################################
# Drop redundant features
# tr_seedlings_


#%%
#######################################################################
# SCALING

x_train_std = tr_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()
x_val_std = val_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()
x_test_std = te_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()

y_train = tr_seedlings['true_height'].copy()
y_val = val_seedlings['true_height'].copy()
y_test = te_seedlings['true_height'].copy()

scaler = MinMaxScaler()
x_train_norm = scaler.fit_transform(x_train_std)
x_val_norm = scaler.transform(x_val_std)
x_test_norm = scaler.transform(x_test_std)

x_train = x_train_norm
x_val = x_val_norm
x_test = x_test_norm

#%%
# EXTRA STUFF

# # save the scaler
# pickle.dump(scaler, open('Height_prediction/models/scaler_final.pkl', 'wb'))

# # To save your model:
# nn_model_filename = 'height_predictor/saved_models/nn_' + str(i) + '.sav'
# pickle.dump(nn_reg, open(nn_model_filename, 'wb'))

# # Load model
# nn_model_filename = 'height_predictor/saved_models/nn_45.sav'
# height_predictor = pickle.load(open(nn_model_filename,"rb"))

# # You can load the scaler in the same way the next time that you need to use it: 
# scaler = pickle.load(open('height_predictor/saved_models/scaler.pkl', 'rb'))
# X_scaled = scaler.transform(X)

#%%#######################################################################
# First pass (all models on default settings, avg over 50 iterations)


scores_df_all_tests = pd.DataFrame(columns=['mae', 'rmse', 'mape', 'r2', 'model'])
test_counter = 0

for i in range(100):
    # Split dataset into test and train
    # x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25)
    # Instantiate model and fit
    
    kn_reg = KNeighborsRegressor(n_neighbors = 5, weights='uniform').fit(x_train, np.ravel(y_train))
    rf_reg = RandomForestRegressor(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
    ab_reg = AdaBoostRegressor(n_estimators = 50, learning_rate = 1,loss='linear').fit(x_train, np.ravel(y_train))
    gbr_reg = GradientBoostingRegressor(n_estimators = 100).fit(x_train, np.ravel(y_train))
    svm_reg = SVR(kernel='rbf', degree=3, C=1).fit(x_train, np.ravel(y_train))
    xgb_reg = XGBRegressor(n_estimators = 100, objective="reg:squarederror", random_state=42, use_label_encoder=False).fit(x_train, np.ravel(y_train))
    nn_reg = MLPRegressor(hidden_layer_sizes=(100), activation ='relu', solver = 'adam', alpha = 0.0001, learning_rate = 'constant', max_iter=1000).fit(x_train, np.ravel(y_train))

    # Make predictions
    predictions_kn = kn_reg.predict(x_val)
    predictions_rf = rf_reg.predict(x_val)
    predictions_ab = ab_reg.predict(x_val)
    predictions_gbr = gbr_reg.predict(x_val)
    predictions_svm = svm_reg.predict(x_val)
    predictions_xgb = xgb_reg.predict(x_val)
    predictions_nn = nn_reg.predict(x_val)

    score_df_kn = regression_scores(y_true=y_val, y_pred=predictions_kn, model='KNeighbours')
    score_df_rf = regression_scores(y_true=y_val, y_pred=predictions_rf, model='Random Forest')
    score_df_ab = regression_scores(y_true=y_val, y_pred=predictions_ab, model='AdaBoost')
    score_df_gbr = regression_scores(y_true=y_val, y_pred=predictions_gbr, model='Gradient Boosting')
    score_df_svm = regression_scores(y_true=y_val, y_pred=predictions_svm, model='SVM')
    score_df_xgb = regression_scores(y_true=y_val, y_pred=predictions_xgb, model='XGBoost')
    score_df_nn = regression_scores(y_true=y_val, y_pred=predictions_nn, model='NN')

    scores_df_all_models = pd.concat([score_df_kn, score_df_rf, score_df_ab,score_df_gbr, score_df_xgb, score_df_svm, score_df_nn]).reset_index(drop=True)
    # scores_df_all_models.loc[df_idx,'feature_list'] = features_test

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

    test_counter += 1

    if test_counter % 200 == 0:

        print(test_counter, ' tests completed') 
    
scores_df_all_tests[['mae', 'rmse', 'mape', 'r2']] = scores_df_all_tests[['mae', 'rmse', 'mape', 'r2']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['mae', 'rmse', 'mape', 'r2', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()

#%%#############################################
#  Investigate results
scores_df_all_tests_avg.sort_values('r2', ascending=False)

# temp=
# temp

#%%
# predictions2 = rf_reg.predict(x_test)
# predictions2 = kn_reg.predict(x_test)
# predictions2 = ab_reg.predict(x_test)
predictions2 = gbr_reg.predict(x_test)

score2= regression_scores(y_true=y_test, y_pred=predictions2, model='knn')
score2

#%%##########################################################
# KNN fine tuning

score_kn = pd.DataFrame(columns=['neighbours','weights','algorithm','p','mae', 'mse', 'mape', 'r2', 'time','model'])

n_neighbors = [2,3,4,5,6,7,8,9,10]
weights = ['distance','uniform']
algorithm = ['auto']#, 'ball_tree', 'kd_tree', 'brute']
ps=[1,2]
for p in ps:
    for a in algorithm:
        for w in weights:
            for n in n_neighbors:
                kn_reg = KNeighborsRegressor(n_neighbors = n, weights=w, algorithm=a, p=p).fit(x_train, np.ravel(y_train))

                start_time = time.time()
                predictions_kn = kn_reg.predict(x_val)
                pred_time_kn = time.time() - start_time

                score_df_kn = regression_scores(y_true=y_val, y_pred=predictions_kn, model='KNeighbours')
                score_df_kn['neighbours']=n
                score_df_kn['weights']=w
                score_df_kn['time']=pred_time_kn
                score_df_kn['algorithm']=a
                score_df_kn['p']=p

                score_kn=score_kn.append(score_df_kn)


#%%#############################################
#  Investigate results

score_kn.sort_values('r2', ascending=False).head(5)



#%%#############################################
# Verifying results on test data

best_n= 9
best_weight='uniform'

knn = KNeighborsRegressor(n_neighbors=best_n, weights=best_weight)
knn_trained = knn.fit(x_train, y_train)

knn_model_filename = 'Height_prediction/models/knn_best.sav'
pickle.dump(knn_trained, open(knn_model_filename, 'wb'))

knn_preds_test= knn_trained.predict(x_test)
knn_score_test= regression_scores(y_true=y_test, y_pred=knn_preds_test, model='KNeighbours')

knn_score_test



#%%##########################################################
# Adaboost fine tuning

score_ada = pd.DataFrame(columns=['mae', 'rmse', 'mape', 'r2', 'time','model'])

N_estimators=[10, 25,50,75,100, 125, 150]
learning_rate = [0.6,0.7,0.75,0.8,0.85,0.9,1,1.05,1.1,1.15,1.2,1.3,1.4]
loss = ['linear','square', 'exponential']
random_state=44
for n in N_estimators:
    for l in loss:
        for lr in learning_rate:
            ab_reg = AdaBoostRegressor(n_estimators = n, learning_rate = lr,loss=l, random_state=44).fit(x_train, np.ravel(y_train))

            start_time = time.time()
            predictions_ab = ab_reg.predict(x_val)
            pred_time_ab = time.time() - start_time

            score_df_ada = regression_scores(y_true=y_val, y_pred=predictions_ab, model='Adaboost reg')
            score_df_ada['learning_rate']=lr
            score_df_ada['loss']=l
            score_df_ada['time']=pred_time_ab
            score_df_ada['n_estimators']=n
            score_ada=score_ada.append(score_df_ada)

            
#%%#############################################
#  Investigate results

score_ada.sort_values('r2', ascending=False).head(5)

#%%

# ab_reg.feature_names_in_

# ab_reg.feature_importances_


# ab_reg.estimators_



#%%#############################################
# Verifying results on test data

# lr = 0.7
# l='linear'
lr = 0.8
l='square'
# 42

ab = AdaBoostRegressor(n_estimators = 50, learning_rate = lr, loss=l, random_state=42).fit(x_train, np.ravel(y_train))
ada_trained = ab.fit(x_train, y_train)

ada_model_filename = 'Height_prediction/models/ada_best.sav'
pickle.dump(ada_trained, open(ada_model_filename, 'wb'))

start_time = time.time()
ada_preds_test= ada_trained.predict(x_test)
pred_time_ab = time.time() - start_time

ada_score_test= regression_scores(y_true=y_test, y_pred=ada_preds_test, model='Adaboost reg')
ada_score_test['Time']=pred_time_ab
ada_score_test


            
            
            

#%%##########################################################
# RandomForest TUNING

score_rf = pd.DataFrame(columns=['max_features','criterion','mae', 'rmse', 'mape', 'r2', 'time','model'])

# max_features = ['auto', 'sqrt','log2']
# crit = ['squared_error','absolute_error','poisson']
# random_state=44
N_estimators=[50,75, 100, 125, 150]
m_features=['auto','sqrt','log2']
crits=['mae','mse','poisson']
for n in N_estimators:
    for m in m_features:
        for c in crits:
            rf_reg = RandomForestRegressor(n_estimators=n, criterion=c, max_features = m, random_state=44).fit(x_train, np.ravel(y_train))

            start_time = time.time()
            predictions_rf= rf_reg.predict(x_val)
            pred_time_rf = time.time() - start_time

            score_df_rf = regression_scores(y_true=y_val, y_pred=predictions_rf, model='Random Forest')      
            score_df_rf['time']=pred_time_rf
            score_df_rf['max_features']=m
            score_df_rf['criterion']=c
            score_df_rf['n_estimators']=n

            score_rf=score_rf.append(score_df_rf)

#%%
score_rf.sort_values('r2', ascending=False).head()
#%%
# rf_reg.n_features_in_
# rf_reg.base_estimator_
# rf_reg.estimators_
# rf_reg.feature_names_in_
#%%

rf_reg = RandomForestRegressor(criterion=c, max_features = mf, random_state=44)
# .fit(x_train, np.ravel(y_train))

#%%#############################################
#  Investigate results
m_features=['auto','sqrt','log2']
crits=['squared_error','absolute_error','poisson']
for m in m_features:
    for c in crits:
        rf_reg = RandomForestRegressor(criterion=c, max_features = m, random_state=44)
        rf_trained = rf_reg.fit(x_train, y_train)
# score_rf.sort_values('r2', ascending=False)

#%%


clist=['squared_error','absolute_error','poisson']
for cree in clist:
    # best_criteria = 'mae'
    best_max_features = 'sqrt'

    rf = RandomForestRegressor( max_features=best_max_features, random_state=44)
    rf_trained = rf.fit(x_train, y_train)


    # rf_model_filename = 'Height_prediction/models/rf_best.sav'
    # pickle.dump(rf_trained, open(rf_model_filename, 'wb'))

    rf_preds_test= rf_trained.predict(x_test)
    rf_score_test= regression_scores(y_true=y_test, y_pred=knn_preds_test, model='Random Forest')

rf_score_test

#%%#############################################
# Verifying results on test data

best_criteria = 'mae'
best_max_features = 'sqrt'

rf = RandomForestRegressor(criterion=best_criteria, max_features=best_max_features, random_state=44)
rf_trained = rf.fit(x_train, y_train)

rf_model_filename = 'Height_prediction/models/rf_best.sav'
pickle.dump(rf_trained, open(rf_model_filename, 'wb'))

rf_preds_test= rf_trained.predict(x_test)
rf_score_test= regression_scores(y_true=y_test, y_pred=knn_preds_test, model='Random Forest')

rf_score_test

#%%


best_criteria = 'mae'
best_max_features = 'sqrt'

crits=['squared_error','absolute_error','poisson']
for c in crits:
    # rf_reg = RandomForestRegressor(criterion=c, max_features = 'auto', random_state=44)
    print(c)

    rf = RandomForestRegressor( max_features=best_max_features , random_state=44)
    rf_trained = rf.fit(x_train, y_train)

    rf_model_filename = 'Height_prediction/models/rf_best.sav'
    pickle.dump(rf_trained, open(rf_model_filename, 'wb'))


    rf_preds_test= rf_trained.predict(x_test)
    rf_score_test= regression_scores(y_true=y_test, y_pred=knn_preds_test, model='Random Forest')

rf_score_test

#%%
# criterion
c
#%%
del c#criterion