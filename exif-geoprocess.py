#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:26:14 2019

@author: David Loibl
"""

from PIL import Image
from PIL import ExifTags as ExifTags
import pandas as pd
import numpy as np
from os import walk


# User-defined variables
input_path  = '/media/loibldav/TOSHIBA EXT/Drone-Data/KIR19/Cognac/Mavic1'
#input_path  = '/media/loibldav/TOSHIBA EXT/Drone-Data/KIR19/Cognac/Mavic2'
output_path = '/home/loibldav/Processing/exif-experiment'
gcp_csv = '/home/loibldav/Processing/exif-experiment/GCPs_Cognac.csv'
verbosity   = 1   # Reporting level


# gcp_coords = [[42.5852, 76.102575]]
max_offset = 0.0005

# Functions
def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

def calc_decdeg_coords(gps_info_array):
    ''' Calculates decimal degree coordinates from EXIF gpsinfo data. '''
    decdeg_coords = gps_info_array[0][0] + gps_info_array[1][0] / 60 + gps_info_array[2][0] / 36000000
    return decdeg_coords


# Process ground control point coordinates
gcp_data = pd.read_csv(gcp_csv, header=0) 
gcp_x_y_data = gcp_data[['Y', 'X', 'name']].copy()
gcp_coords = gcp_x_y_data.values.tolist()

# Read files recursively
(_, _, filenames) = next(walk(input_path))
filenames = sorted(filenames) #.sort()


print('Found '+ str(len(filenames)) + ' in input directory.')


# 103 degree, 41 minute, 1052 centisecond, 103+41/60+1052/(3600*100)

exif_coords = []
lat_coords  = []
lon_coords  = []

file_counter = 1
for filename in filenames:

    exif = get_exif(input_path + '/' + filename)
    # print(exif)
    
    gpsinfo = {}
    for key in exif['GPSInfo'].keys():
        decode = ExifTags.GPSTAGS.get(key,key)
        gpsinfo[decode] = exif['GPSInfo'][key]
    # print(gpsinfo)
    
    latitude = calc_decdeg_coords(gpsinfo['GPSLatitude']) # [0][0] + gpsinfo['GPSLatitude'][1][0] / 60 + gpsinfo['GPSLatitude'][2][0] / 360000
    longitude = calc_decdeg_coords(gpsinfo['GPSLongitude']) # [0][0] + gpsinfo['GPSLongitude'][1][0] / 60 + gpsinfo['GPSLongitude'][2][0] / 360000
    
    exif_coords.append([filename, latitude, longitude])
    lat_coords.append(latitude)
    lon_coords.append(longitude)
    
    
    # print('Latitude: '+ str(latitude))
    # print('Longitude: '+ str(longitude))
   
    # data = pd.read_csv(input_path + '/' + filename, header=0) # , usecols=['LS_DATE','SC_median']
    
    if verbosity >= 1:
        print('Working on ' + filename + ' ('+ str(file_counter) +' of '+ str(len(filenames)) + ') ...', end='\r', flush=True)
    '''    
    # print(str, end='\r')
        # sys.stdout.flush()
    if verbosity >= 2:
        print(data.shape)
        print(data.size)
        
    '''
    file_counter += 1

print('\nFinished EXIF input data processing\n')

result_df_labels = ['filename', 'lat_gps', 'lon_gps', 'gcp_name', 'lat_gcp', 'lon_gcp', 'file_path']
result_df_full = pd.DataFrame(columns=result_df_labels)

for gcp_coord in gcp_coords:
    print('GPS coord: '+ str(gcp_coord))
    lat_coord_np = np.array(lat_coords)
    lat_w1 = np.where(lat_coord_np < (gcp_coord[0] + max_offset))[0]
    lat_w2 = np.where(lat_coord_np > (gcp_coord[0] - max_offset))[0]
    lat_w_isin = np.isin(lat_w1, lat_w2)
    lat_w_isin_ids = lat_w1[lat_w_isin]
    
    lon_coord_np = np.array(lon_coords)
    lon_w1 = np.where(lon_coord_np < (gcp_coord[1] + max_offset))[0]
    lon_w2 = np.where(lon_coord_np > (gcp_coord[1] - max_offset))[0]
    lon_w_isin = np.isin(lon_w1, lon_w2)
    lon_w_isin_ids = lon_w1[lon_w_isin]
    
    print('Found '+ str(len(lat_w_isin)) +' images fitting in latitude')
    print('Found '+ str(len(lon_w_isin)) +' images fitting in longitude')
    
    w_match = np.isin(lat_w_isin_ids, lon_w_isin_ids)
    match_ids = lat_w_isin_ids[w_match]
    
    #print('Latitude Ids:'+ str(lat_w_isin_ids))
    #print('Longitude Ids:'+ str(lon_w_isin_ids))
    
    results_data = [exif_coords[i] for i in match_ids]
    match_files = [item[0] for item in results_data]
    print('Result datasets: '+ str(match_files) +'\n')
    
    result_df = pd.DataFrame.from_records(results_data, columns=['filename', 'lat_gps', 'lon_gps'])
    result_df = result_df.assign(gcp_name=gcp_coord[2], lat_gcp=gcp_coord[0], lon_gcp=gcp_coord[1], file_path=input_path)
    result_df_full = result_df_full.append(result_df, ignore_index=True)

result_df_full.to_csv(output_path +'/gcp_images.csv', index=False)