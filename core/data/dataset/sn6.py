"""SpaceNet6 Dataset."""
import os
import torch
import random
import re
import cv2
import csv
import numpy as np
import pandas as pd
import json
import rasterio as rio
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from imantics import Polygons, Mask
from scipy.ndimage.morphology import distance_transform_edt
import glob

class SpaceNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_dir,
        image_dir,
        sar_dir,
#         test_dir,
        transform=None,
    ):
        """
        Args:
            image_dir (string): Path to RGB images directory.
            sar_dir (string): ...
            mask_dir (string): ... 
            tile_number (string): ...
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_dir = csv_dir
        self.data = pd.read_csv(csv_dir)
        self.image_dir = image_dir
        self.sar_dir = sar_dir
        self.mask_list = self.create_poly_list()
        self.transform = transform
        self.tile_id_list = self.tile_id_list()
#         self.test_dir = test_dir
#         self.test_id_list = self.test_id_list()
        
    def __len__(self):
        return len(self.tile_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = self.get_rgb(idx)
        mask = self.get_mask(idx)
        sar = self.get_sar(idx)
        edge = self.get_edge(idx, mask)
        
        if self.transform:
            image, mask, _ = self.transform(image, mask)

        return sar, mask, image, edge 
    
    def get_sar(self, idx):
        create_SAR_path = self.sar_dir +"SN6_Train_AOI_11_Rotterdam_SAR-Intensity_" + self.tile_id_list[idx][0] + ".tif"
        with rio.open(create_SAR_path) as lidar_dem:
            img = np.zeros((900,900,4))
            img[:,:,0] = lidar_dem.read(1)
            img[:,:,1] = lidar_dem.read(2)
            img[:,:,2] = lidar_dem.read(3)
            img[:,:,3] = lidar_dem.read(4)
        return torch.from_numpy(img*255/np.max(img))
    
    def get_rgb(self, idx):
        create_RGB_path = self.image_dir +"SN6_Train_AOI_11_Rotterdam_PS-RGB_" + self.tile_id_list[idx][0] + ".tif"
        with rio.open(create_RGB_path) as lidar_dem:
            img = np.zeros((900,900,3))
            img[:,:,0] = lidar_dem.read(1)
            img[:,:,1] = lidar_dem.read(2)
            img[:,:,2] = lidar_dem.read(3)
        return torch.from_numpy(img)

    def get_mask(self, idx):
        return self.generate_mask(self.mask_list[idx][1], self.mask_list[idx][2])
         
    def generate_mask(self, start, end):
        mask_img = Image.new('1', (900, 900), 0)
        poly = ImageDraw.Draw(mask_img)
        for i in range(start,end+1):
            row = self.data.loc[i,'PolygonWKT_Pix']
            expression = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", row)
            tup = (float(expression[0]), float(expression[1]))
            for i in range(2,len(expression),2):
                temp = (float(expression[i]),float(expression[i+1]))
                tup = tup + temp
            poly.polygon(tup, outline = 1, fill = 1)
        mask = np.array(mask_img)
        mask = torch.from_numpy(mask).float()
        return mask 
    
    def onehot_to_binary_edges(self, mask, radius):
        mask = mask.numpy()    
        if radius < 0:
            return mask
        mask = mask.astype(np.uint8)
        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        edgemap = np.zeros(mask.shape)
        dist = distance_transform_edt(mask_pad)
        #print(dist)
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
#         edgemap = np.expand_dims(edgemap, axis=0)    
#         edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap
    
    def get_edge(self, idx, mask):
        _edgemap = mask
        _edgemap = self.onehot_to_binary_edges(_edgemap, 1)
        edgemap = torch.from_numpy(_edgemap).float()
        return edgemap
    
    def create_poly_list(self):
        pointer = 0
        poly_index_list = []
        while pointer < len(self.data):
            start = pointer
            end = pointer
            tile_number_st = self.data['ImageId'][pointer].rfind('_')
            tile_number = self.data['ImageId'][pointer][tile_number_st+1:]
            while (pointer+2) < len(self.data) and (self.data['TileBuildingId'][pointer] < self.data['TileBuildingId'][pointer+1]):
                end +=1
                pointer +=1
            poly_index_list.append([tile_number, start, end, self.data['ImageId'][pointer]])
            pointer +=1
        return poly_index_list
        
    def random_rotation(self, image, mask, sar):
        orient = np.random.randint(0, 4)
        image = np.rot90(image, orient)
        mask = np.rot90(mask, orient)
        sar = np.rot90(sar, orient)
        return image, mask, sar, orient

    def tile_id_list(self):
        tile_list = []
        csv_list = self.create_poly_list()
        for tile in range(len(csv_list)):
            tile_list.append([csv_list[tile][3], csv_list[tile][0]])
        return tile_list
    
    def create_output_csv(self):
        with open('solution.csv', 'w') as csv_file:
            fieldnames = ['ImageId','PolygonWKT_Pix','Confidence']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    
    def csv_add_newline(self, image_id, polygons, score):
        with open('solution.csv', 'a', newline='') as csv_file:
            fieldnames = ['ImageId','PolygonWKT_Pix','Confidence']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'ImageId': image_id, 'PolygonWKT_Pix': polygons, 'Confidence': score})
    
    def create_polygon(self, idx, mask):
        mask = mask.numpy()
        binary_mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range (len(contours)):
            points = []
            contour = np.squeeze(contours[i])
            for j in range(len(contour)):
                points.append((contour[j][0], contour[j][1]))
            if contour.shape[0] >= 3:
                polygon = Polygon(points)   
            #add a newline to csv!   
            image_id = self.mask_list[idx][3]
            self.csv_add_newline(image_id, polygon.wkt, 1.0)
            #print(polygon.wkt)