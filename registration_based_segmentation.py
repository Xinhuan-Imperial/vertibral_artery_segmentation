#!/usr/bin/python
# -*- coding: UTF-8 -*-
import SimpleITK as sitk
import os,time,math,cv2
import numpy as np
from ctypes import *

if __name__=="__main__":
    start=time.time()
    filedir=r'E:/sub2/artery/co2/'
    file_list=os.listdir(filedir)
    #CO2:120,resting:80
    venc=120
    name_list=[]
    for i in file_list:
        tmp0=os.path.splitext(i)[0]
        tmp1=os.path.splitext(i)[1]
        if tmp1=='.img' and tmp0[0:3]=='ana':#get the file name of all anatomic images
            name_list.append(i)
    #calculate the magnitude image of Dynamic 1
    anatomic_name=r'E:/sub2/artery/co2/anatomic_0001.img'
    velocity_name=r'E:/sub2/artery/co2/velocity_0001.img'
    anatomic_image=sitk.ReadImage(anatomic_name,sitk.sitkFloat32)
    anatomic=sitk.GetArrayFromImage(anatomic_image)
    velocity_image=sitk.ReadImage(velocity_name,sitk.sitkFloat32)
    velocity=sitk.GetArrayFromImage(velocity_image)
    angle=velocity/venc*math.pi
    Imoving=2*anatomic*np.sin(angle)   
    Imoving=Imoving.squeeze()
    image_size=Imoving.shape
    Imoving=Imoving.ravel()
    
    mask_refname=r'E:/sub2/artery/mask.img'#this is the mask of Dynamic 1
    mask_ref=sitk.ReadImage(mask_refname,sitk.sitkFloat32)
    mask_ref=sitk.GetArrayFromImage(mask_ref)
    mask_ref=mask_ref.squeeze()
    mask_ref=mask_ref.ravel()

    N=image_size[0]*image_size[1]
    spacing_x=16#gid spacing of the B-spline control knots
    spacing_y=16
    num_row=image_size[0]
    NUM_PARAMS=int(2*(num_row/spacing_y+4)*(N/num_row/spacing_x+4))
    penalty=0.5

    ffd_penal_registration_gpu2d=np.ctypeslib.load_library(r'./ffd_penal_registration_gpu2d.dll','.')
    ffd_penal_registration_gpu2d.make_init_knots.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),c_int,c_int,c_int,c_int,c_int]
    ffd_penal_registration_gpu2d.Steepest_Descent_solver.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),c_int,c_int,c_int,c_int,c_int,c_float]
    ffd_transform_gpu2d=np.ctypeslib.load_library(r'./ffd_transform_gpu2d.dll','.')
    ffd_transform_gpu2d.conduct_ffd_transform.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), c_int, c_int, c_int, c_int, c_int]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter()
    writer.open('registered.avi', fourcc, 30.0, image_size, isColor=False)
    writer2 = cv2.VideoWriter()
    writer2.open('moving.avi', fourcc, 30.0, (image_size[0]*2, image_size[1]), isColor=False)

    for i in name_list:
        tmp1=os.path.join(filedir,i)
        anatomic_img = sitk.ReadImage(tmp1,sitk.sitkFloat32)
        tmp2=tmp1.replace('anatomic','velocity')
        velocity_img=sitk.ReadImage(tmp2,sitk.sitkFloat32)

        velocity=sitk.GetArrayFromImage(velocity_img)
        anatomic=sitk.GetArrayFromImage(anatomic_img)
        angle=velocity/venc*math.pi
        Iref=2*anatomic*np.sin(angle)
        Iref=Iref.squeeze()
        tmp = cv2.normalize(Iref, None, 0,255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        h_x = np.zeros(NUM_PARAMS,dtype=np.float32)
        Iref=Iref.ravel()     
        h_out=np.zeros(N,dtype=np.float32)
        ffd_penal_registration_gpu2d.make_init_knots(h_x,spacing_x,spacing_y,N,num_row,NUM_PARAMS)
        ffd_penal_registration_gpu2d.Steepest_Descent_solver(Imoving,Iref, h_x, h_out, N, num_row, NUM_PARAMS,spacing_x,spacing_y,penalty)
        h_out=np.reshape(h_out,(int(N/num_row),int(num_row)))/np.max(h_out)
        h_out = cv2.normalize(h_out, None, 0,255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #normalize for visualization in openCV
        writer.write(h_out)
                
        mask_transformed = np.zeros(N, dtype=np.float32)   
        ffd_transform_gpu2d.conduct_ffd_transform(mask_ref,h_x,mask_transformed,N,num_row,NUM_PARAMS,spacing_x,spacing_y)#this time Iref will be transformed and the transformed image is h_out
        mask_transformed = np.reshape(mask_transformed, (int(N / num_row), int(num_row)))/np.max(mask_transformed)
        mask_transformed = cv2.normalize(mask_transformed, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1) #normalize for visualization in openCV
        tmp1=np.concatenate((tmp,mask_transformed),axis=1)
        writer2.write(tmp1)

    writer.release()
    writer2.release()
    cv2.destroyAllWindows()
    end=time.time()
    print('Total time collapsed: ',end-start)
