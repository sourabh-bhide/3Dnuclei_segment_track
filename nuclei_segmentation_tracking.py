#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:46:34 2021

@author: sourabh.j.bhide@gmail.com

Usage : nuclei_segmentation_tracking --config "/Path_to_config/configuration.json"
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import h5py
import os
import pandas as pd

import argparse
import logging
import json

from glob import glob
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from skimage.transform import rescale, resize
from skimage.measure import regionprops_table
from stardist import random_label_cmap
from stardist.models import StarDist3D


np.random.seed(6)
lbl_cmap = random_label_cmap()

#import gputools

import trackpy as tp

'''
config = '/Volumes/Groupdir/SUN-DAN-Semb/Sourabh/h2btdtomatongn3gfp/configuration_airflow.json'

'''
#%%
#functions
def read_config(config_filename):
    #reads a json config file
    with open(config_filename) as json_file:
        config_data = json.load(json_file)       
        
    file_name = config_data['filename']
    target_shape = config_data['output_shape']
    stardist_model_dir = config_data['model_dir']            
    return file_name, target_shape, stardist_model_dir 

def labels_to_regionprops(label_image, intensity_image):
    #extracts region properties from instance segmentation results
    props = regionprops_table(label_image,intensity_image=intensity_image, properties=['centroid','area','minor_axis_length','major_axis_length','label'])     
    props['aspect_ratio'] = props['major_axis_length'] / props['minor_axis_length']
    info_df = pd.DataFrame(props)
    return info_df

def stardist_prediction(img, model, axis_norm=(0,1,2),file_key='h2btdtomatongn3gfp'):
    #returns instance segmentation using stardist model and region props measurement 
    #normalize and predict
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm, n_tiles=(2,4,4))
    #logging.info('')
    #Measure region properties

    regionprops_image = labels_to_regionprops(labels, img)
    return labels, regionprops_image

def track_regionprops(df):
    #returns dataframe with tracking results using trackpy
    #read regionprops
    logging.error('Regionprops dataframe not found')
    #link timepoints
    tdf = tp.link_df(df,25,t_column='t',pos_columns=['centroid-0','centroid-1','centroid-2'],memory=5)
    return tdf

def run_full_analysis(config_path):
    
    #1. Read input config files   
    filename, output_shape, model_dir = read_config(config_path) 

    #make directory for storing timepoints
    base_dir, __ = os.path.split(filename)
    file_key, extension = os.path.splitext(__)
    
    timepoints_dir = os.path.join(base_dir,'timepoints')
    os.makedirs(timepoints_dir, exist_ok=True)
    
    labels_dir = os.path.join(base_dir,'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    #2. Create log file
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename= os.path.join(base_dir , file_key+'.log') , mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    
    logging.debug(extension)
    
    #3. Load 4D image data
    if os.path.exists(filename): 
        logging.info('%s file found',filename)
    else:
        logging.error('%s file not found. Please check your configuration for correct file path',filename)
        
    #open hdf5 file
    if extension == '.h5' : 
        f = h5py.File(filename, 'r')
        logging.info(f[file_key]['ImageData'].keys())
        raw = f.get(file_key+'/ImageData/Image')
        n_timepoints = raw.shape[1]
        logging.info('%s timepoints found',n_timepoints)
    
    #open tif file
    elif extension == '.tif' :
        raw = imread(filename)
        logging.info('the tif file has been read')
        n_timepoints = raw.shape[0]
        logging.info('%s timepoints found',n_timepoints)
        
    else : logging.error('Cannot recognise file format. File should be h5 ot tif')
    
    logging.info('Data has been read')
    
    #4. Define Stardist model for predicting labels
    model = StarDist3D(None, name='stardist', basedir=model_dir)
        
    #5. Run predictions; save raw images, labels, and region props
    region_props = pd.DataFrame()  
    for i in range(n_timepoints):
        logging.info('Processing timepoint : '+str(i))
        
        #resizing images
        if extension == '.h5'  : tp_resize = resize(raw[0,i,:,:,:], output_shape=output_shape , order=0)
        elif extension == '.tif' : tp_resize = resize(raw[i,:,:,:], output_shape=output_shape , order=0)
        
        #predicting labes using stardist model
        labels, regionprops_image = stardist_prediction(tp_resize, model, file_key=file_key)
        
        save_tiff_imagej_compatible(timepoints_dir+'/'+file_key+'_'+str(i)+'_isotropic.tif', tp_resize, axes='ZYX')
        logging.debug('raw file saved')
        
        save_tiff_imagej_compatible(labels_dir+'/'+file_key+'_'+str(i)+'_isotropic_labels.tif', labels, axes='ZYX')
        logging.debug('label file saved')
        
        region_props = region_props.append(regionprops_image)
        logging.debug('region props appended')
        
        logging.info('Processing done for timepoint: %s', i)
      
    logging.info('All timepoint has been converted to tif format and segmented')
    region_props.to_csv(os.path.join(base_dir,file_key+'_regionprops.csv'))
    
    #6. Tracking
    tracks_df = track_regionprops(region_props)
    logging.info('Tracking initiated')
    tracks_df.to_csv(os.path.join(base_dir,file_key+'_regionprops_trackpy.csv'),index=False)
    logging.info('Tracking is complete')
    

#%%

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="configuration files")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    configuration_file_path = args.config
    
    run_full_analysis(configuration_file_path)
    
    
    
    
    
    
    