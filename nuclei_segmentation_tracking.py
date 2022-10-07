#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:46:34 2021

@author: sourabh.j.bhide@gmail.com
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
import os
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
import logging
import json

from glob import glob
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from skimage.transform import rescale, resize
from stardist import random_label_cmap
from stardist.models import StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()

#import gputools

import trackpy as tp

def read_config(config_filename):
    with open(config_filename) as json_file:
        config_data = json.load(json_file)  
    return config_data

config = read_config('/path/to/your/config/file/configuration.json')

#%%
#data preparation

def bytescaling(im):
    return im

def data_preparation(config):
    #read input files    
    filename = config['filename']
    output_shape = config['output_shape']

    #make directory for storing timepoints
    base_dir, __ = os.path.split(filename)
    file_key, extension = os.path.splitext(__)
    
    timepoints_dir = os.path.join(base_dir,'timepoints')
    os.makedirs(timepoints_dir, exist_ok=True)   
    
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename= os.path.join(base_dir , file_key+'.log') , mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    
    logging.info(extension)
    #open hdf5 file
    if extension == '.h5' : 
        f = h5py.File(filename, 'r')
        logging.info(f[file_key]['ImageData'].keys())
        raw = f.get(file_key+'/ImageData/Image')
        n_timepoints = raw.shape[1]
        logging.info('%s timepoints found',n_timepoints)
        
    elif extension == '.tif' :
        raw = imread(filename)
        logging.info('the tif file has been read')
        n_timepoints = raw.shape[0]
        logging.info('%s timepoints found',n_timepoints)
        
    else : logging.error('Cannot recognise file format. File should be h5 ot tif')
    logging.info('All timepoint has been converted to tif format')        
  
    #transform and store timpoints
    for i in range(n_timepoints):
        logging.info('Processing timepoint : '+str(i))
        if extension == '.h5' : tp = bytescaling(raw[0,i,:,:,:])
        elif extension == '.tif' : tp = bytescaling(raw[i,:,:,:])
        
        tp_resize = resize(tp, output_shape=output_shape , order=0)
        save_tiff_imagej_compatible(timepoints_dir+'/'+file_key+'_'+str(i)+'_isotropic.tif', tp_resize, axes='ZYX')
        logging.info('Done')
      
    logging.info('All timepoint has been converted to tif format')
    
#%%

data_preparation(config)

#%%
#prediction

def labels_to_regionprops(label_image, image_name, intensity_image):
    props = measure.regionprops_table(label,intensity_image=intensity_image properties=['centroid','area','minor_axis_length','major_axis_length','label'])     
    props['image_name'] = image_name
    props['aspect_ratio'] = props['major_axis_length'] / props['minor_axis_length']
    info_df = pd.DataFrame(props)
    return info_df
    
def stardist_prediction(config):
    #read input files    
    filename = config['filename']
    model_dir = config['model_dir']
    
    #make directory for storing timepoints
    base_dir, __ = os.path.split(filename)
    file_key, extension = os.path.splitext(__)
    timepoints_dir = os.path.join(base_dir,'timepoints')
    os.makedirs(timepoints_dir, exist_ok=True) 
    
    labels_dir = os.path.join(base_dir,'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    #create logging files
    
    model = StarDist3D(None, name='stardist', basedir=model_dir)
    n_channel = 1 
    axis_norm = (0,1,2) 
    n_timepoints = len(os.listdir(timepoints_dir))
    
    region_props = pd.DataFrame()
    
    for i in range(n_timepoints):
        
        logging.info('Predicting instances for timepoint: %s', i)
        
        img = imread(os.path.join(timepoints_dir, file_key+'_'+str(i)+'_isotropic.tif'))
        img_norm = normalize(img, 1,99.8, axis=axis_norm)
        labels, details = model.predict_instances(img_norm)#, n_tiles=(2,4,4))

        logging.info('writing the image')
        save_tiff_imagej_compatible(os.path.join(labels_dir+'/'+file_key+'_'+str(i)+'_isotropic_labels.tif'), labels, axes='ZYX')
        
        #Measure region properties
        img_name = file_key+'_'+str(i)+'_isotropic.tif'
        regionprops_image = labels_to_regionprops(labels, img_name, img)
        region_props = region_props.append(regionprops_image)
    
    region_props.to_csv(os.path.join(base_dir,file_key+'_regionprops.csv'))
    logging.info('Segmentation predictions for all timepoints are complete')
    
stardist_prediction(config)

#%%
#tracking

def track_regionprops(config):
    #read input files 
    filename = config['filename']
    model_dir = config['model_dir']
    
    #make directory for storing timepoints
    base_dir, __ = os.path.split(filename)
    file_key, extension = os.path.splitext(__)    
    
    #read regionprops
    df = pd.read_csv(os.path.join(base_dir,file_key+'_regionprops.csv'))
    logging.error('Regionprops dataframe not found')
    
    #link timepoints
    tdf = tp.link_df(df,25,t_column='t',pos_columns=['centroid-0','centroid-1','centroid-2'],memory=5)
    tdf.to_csv(os.path.join(base_dir,file_key+'_regionprops_trackpy.csv'),index=False)
    logging.info('Tracking is complete')




    
    
    
    
    
