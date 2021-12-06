#%%
# Import libraries
#######################################################################

import os
import glob
import time
from zipfile import ZipFile
import random
import datetime
import shutil
import gc

import numpy as np
from numpy.core.numeric import ones
import pandas as pd
from math import dist
from sklearn import neighborsx
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
matplotlib.rcParams.update({'font.family': 'Times New Roman'})
import seaborn as sns
from skimage.feature import peak_local_max
from skimage import img_as_ubyte

import cv2
import albumentations as A
import torch


from deepforest import main
from deepforest import preprocess
from deepforest import utilities
from deepforest import visualize
from deepforest import evaluate

from pytorch_lightning.loggers import CSVLogger
from pandas_profiling import ProfileReport

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
    """Randomly split available data into train, validation and test data sets

    Returns: 
        Annotation data frames for train, validation and terst data sets.
    """

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

    # Unzip annotations file
    path = "CVAT/" + file_name
    with ZipFile(path, 'r') as zip:
        zip.extractall()

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



#######################################################################
# Obtain annotations relating to a specific image tile
#######################################################################
def tile_annotations (windowsO, annotations):
    """Obtains the annotations relating to s specific image tile/window


    Returns:
        annotations dataframe
    """
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


#######################################################################
# Obtain annotations relating to a specific image tile
#######################################################################
def plot_predictions_from_df(df, img, colour = (255, 255, 0)):
    """ Plot the predicted bounding boxes

    Returns:
        An image containing the original image as well as the predicted bounding boxes
    """
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

    # path to store aumented images
    augmented_path=dir 

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

    images_to_generate=number  
    i=1                      

    if augs != 'none':
        while i<=images_to_generate:
            image=random.choice(image_list)
            original_image = cv2.imread(image)
            original_annotations= annotations.loc[(annotations.image_path == os.path.basename(image)),('xmin', 'ymin','xmax','ymax','label')]

            # perform transformation
            transformed = transform(image=original_image, bboxes=original_annotations.values.tolist())#,min_visibility=0.8)

            image_name=os.path.splitext(os.path.basename(image))[0]
            new_image_path= "%s%s_augmented_%s.png" %(augmented_path,image_name, i)
            transformed_image = img_as_ubyte(transformed['image'])  #Convert an image to unsigned byte format, with values in [0, 255].
            cv2.imwrite(new_image_path, transformed_image) # save transformed image to path

            augmented_annotations = pd.DataFrame(transformed['bboxes'], columns=('xmin', 'ymin','xmax','ymax','label'), dtype='int')
            augmented_annotations.insert(loc=0, column='image_path', value=os.path.basename(new_image_path))
            framesA=[annotations, augmented_annotations]
            annotations = pd.concat(framesA)

            i += 1  

    return annotations

#######################################################################
# Save the annotations to the specified directory
#######################################################################
def save_annotations (annotations, dir):
    """ Save the annotations to the specified directory as a .csv file and return the directory

    Returns:
        The directory of of the saved annotations .csv file
    """
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
    """ Configure the DeepForest algorithm/model

    Returns:
        The configured model
    """
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

    logger=CSVLogger('logs',name=file_name)
    
    #create a pytorch lighting trainer used to training 
    m.create_trainer(logger=logger)

    # Uncomment the line below if pytorch logging is required instead of .csv logging
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
    """Evaluate the trained DeepForest model

    Returns:
        The results of the evaluation
    """
    save_dir = os.path.join(os.getcwd(),'Results')
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    results = m.evaluate(annotations_file, os.path.dirname(annotations_file), savedir= save_dir, iou_threshold = thresh)

    return results

#######################################################################
# Cleaning files used for previous training/testing iterations
#######################################################################
def clean_training():
    """Delete training images from previous iterations

    Returns:
        None
    """ 
    # Empty training folder between iterations
    train_files = glob.glob('Train/*.png')
    for f in train_files:
        os.remove(f)


def clean_test():
    """Delete testing images from previous iterations

    Returns:
        None
    """ 
    # Empty training folder between iterations
    test_files = glob.glob('Test/*.tif')
    for f in test_files:
        os.remove(f)


def clean_annotations ():
    """Delete annotations from previous iterations

    Returns:
        None
    """
    # Empty training folder between iterations
    annotations_files = glob.glob('Annotations/*')
    for f in annotations_files:
        os.remove(f)


def clean_tiles ():
    """Delete tiles images from previous iterations

    Returns:
        None
    """
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
    """Delete augmented images

    Returns:
        None
    """
    augmented_files = glob.glob('Train/*augmented*.png')
    for f in augmented_files:
        os.remove(f)

    augmented_files = glob.glob('Val/*augmented*.png')
    for f in augmented_files:
        os.remove(f)

#######################################################################
# Local Maxima: Find
#######################################################################
def lm (dtm, distance, min_height, offset, max_peaks):
    """Searches the DTM for the local maxima

    Returns:
        coordinates and height of local maximas as a pandas dataframe
    """
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
    y_shift = ystart-5 + row_shift*0.5
    s=0

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
            # Calculate distance to available neighbours
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
    """Read ground-truth heights from .csv file and appent this to the data frame

    Returns:
        Data frame with the ground truth heights appended
    """
    true_height=pd.read_excel(path, index_col=0)

    dfFinal= pd.merge(boxes,true_height,left_on = 'position', right_on='number')
    dfFinal['errorH'] = dfFinal['Height'] - dfFinal['true_height']
    dfFinal=dfFinal.sort_values(by='position', ascending=True).reset_index(drop=True)
    return dfFinal

#######################################################################
# Extract seedling crops and save
#######################################################################
def seedling_extract (dfFinal, imgO, seedling_dir=None, tuning=True):
    """Extract and save images of the detected seedlings

    Returns:
        - Data frame of bounding box positions
        - Cropped images
    """
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
    """Extract features relating to the detected seedlings

    Returns:
        - Data frame of detected seedlings with all extracted features
    """
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
    """Determine neighbouring seedlings based on a seedling's location within a tray

    Returns:
        - List of neighbouring seedlings
    """
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

    score_df = pd.DataFrame(columns=['model', 'mae', 'rmse', 'r2', 'features', 'feature_list_idx'])

    score_df['features'] = score_df['features'].astype(object)
    score_df.loc[0,'model'] = model
    score_df.loc[0,'mae'] = mean_absolute_error(y_true, y_pred)
    score_df.loc[0,'rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
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
validation_iou_threshold = 0.4


epoch=15

lr_schedules = ['default' ,'StepLR', 'reduce_on_plat', 'exponential']
optimisers = ['sgd', 'Adadelta', 'Adam', 'Rprop']
train_learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [1,2,3,4,5,6]


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

results_df.to_csv('Results/DeepForest_results/Hyperparameter_tuning_results_lr_0_0001.csv', index=False)


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
Augmentations = ['all', 'SSR','HF','RBC','RR']
# Augmentations = ['HF_RBC']

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

model_pathF='DF_models/Report models/Hyper models/Augs__aug=none_minVis=2_epoch=4___run3.pt'

m = main.deepforest()
m.model.load_state_dict(torch.load(model_pathF))


patch_overlaps=[0.1,0.2,0.25,0.3]
patch_sizes=[900,925,950,975,1000]
nms_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
score_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]


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

                val_boxes=pd.DataFrame()

                for i in val_annotations.image_path.unique():
                    start_time = time.time()
                    img ,box = m.predict_tile(os.path.join('Agisoft/Ortho/val/',i),return_plot=True,patch_overlap=patch_overlap,patch_size=patch_size)
                    prediction_time = time.time() - start_time
                    box['image_path'] = i
                    val_boxes=val_boxes.append(box)

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
#%%
######################################################
# LOCAL MAXIMA - TUNING
######################################################

# Load the chosen model
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

patch_overlap=0.1
patch_size=900
m.config["nms_thresh"] =0.05
m.config["score_thresh"] =0.3

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

lm_results.to_csv('Results/LM_results/LM_tuning_results_v3.csv', index=False)



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


#%%#######################################################################
# MODELLING
#######################################################################


#######################################################################
# SCALING

x_train_std = tr_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()
x_val_std = val_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()
x_test_std = te_seedlings.drop(columns=['tray','position','true_height','errorH']).copy()

y_train = tr_seedlings['true_height'].copy()
y_val = val_seedlings['true_height'].copy()
y_test = te_seedlings['true_height'].copy()

scaler = MinMaxScaler()
# save the scaler
pickle.dump(scaler, open('Height_prediction/models/scaler_final.pkl', 'wb'))

x_train_norm = scaler.fit_transform(x_train_std)
x_val_norm = scaler.transform(x_val_std)
x_test_norm = scaler.transform(x_test_std)

x_train = x_train_norm
x_val = x_val_norm
x_test = x_test_norm



#%%#######################################################################
# First pass (all models on default settings, avg over 50 iterations)
scores_df_all_tests = pd.DataFrame(columns=['mae', 'rmse', 'r2', 'model'])
test_counter = 0

for i in range(100):
    # Split dataset into test and train
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

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)
    
scores_df_all_tests[['mae', 'rmse', 'r2']] = scores_df_all_tests[['mae', 'rmse', 'r2']].astype(float)
scores_df_all_tests_summary = scores_df_all_tests[['mae', 'rmse', 'r2', 'model']]
scores_df_all_tests_summary = scores_df_all_tests_summary.reset_index(drop=True)
scores_df_all_tests_mean = scores_df_all_tests_summary.groupby(['model'], as_index=False).mean()


#%%##########################################################
# KNN fine tuning

score_kn = pd.DataFrame(columns=['neighbours','weights','algorithm','p','mae', 'mse', 'r2', 'time','model'])

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


#%%##########################################################
# Adaboost fine tuning

score_ada = pd.DataFrame(columns=['mae', 'rmse', 'r2', 'time','model'])

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


#%%##########################################################
# RandomForest TUNING

score_rf = pd.DataFrame(columns=['max_features','criterion','mae', 'rmse', 'r2', 'time','model'])

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
