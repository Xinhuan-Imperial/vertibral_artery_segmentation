#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os,time,math,cv2
import numpy as np
from skimage import measure
import SimpleITK as sitk

def adaptive_region_grow(mask_pre,magnitude):
    labels = measure.label(mask_pre)
    jj = measure.regionprops(labels)
    num_vessels=len(jj)
    seeds=np.zeros((num_vessels,2),dtype=np.int16)#the number of connected regions in mask_pre, as the seed of next frame
    #get the centroid of each connected region, which will be the seeds of region grow of the next frame
    for i in range(num_vessels):
        tmp=np.round(jj[i].centroid)
        seeds[i,:]=tmp
    M,N=magnitude.shape[0],magnitude.shape[1]
    J=np.zeros((M,N),dtype=np.uint8)
    fc=np.zeros((num_vessels,1),dtype=np.float32)
    fc_sum=np.zeros((num_vessels,1),dtype=np.float32)
    fc_count=np.zeros((num_vessels,1),dtype=np.float32)
    for i in range(num_vessels):
        x=seeds[i,1]
        y=seeds[i,0]
        m=magnitude[y,x]
        fc_sum[i]=m
        fc_count[i]=1
        fc[i]=fc_sum[i]/fc_count[i]#average pixel intensity in the masked region
        J[y,x]=1+i

    while True:
        count1=np.sum(J)
        for i in range(0,M):
            for j in range(0,N):
                index=J[i,j]#each region has its own threshold, so labelled with different color
                if index!=0:
                    if ((i-1)>=0)&((i+1)<M)&((j-1)>=0)&((j+1)<N):
                        for u in [-1,0,1]:
                            for  v in [-1,0,1]:                              
                                if (J[i+u,j+v]==0)&(magnitude[i+u,j+v]>0.7*fc[index-1]):#region growth & use mean as the referance
                                    J[i+u,j+v]=index
                                    fc_sum[index-1]+=magnitude[i+u,j+v]
                                    fc_count[index-1]+=1
                                    fc[index-1]=fc_sum[index-1]/fc_count[index-1]
        count2=np.sum(J)
        if count1==count2:
            break
    return J

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
    magnitude=2*anatomic*np.sin(angle)   
    magnitude=magnitude.squeeze()
    image_size=magnitude.shape

    #this is the mask of Dynamic 1
    mask_name=r'E:/sub2/artery/mask.img'
    mask_pre=sitk.ReadImage(mask_name,sitk.sitkFloat32)
    mask_pre=sitk.GetArrayFromImage(mask_pre)
    mask_pre=mask_pre.squeeze()

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
        magnitude=2*anatomic*np.sin(angle)
        magnitude=magnitude.squeeze()
        tmp = cv2.normalize(magnitude, None, 0,255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)#normalize for visualization in openCV
                
        mask =adaptive_region_grow(mask_pre,magnitude)
        mask=(mask>0).astype(np.uint8)
        mask_normalized = cv2.normalize(mask, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1) #normalize for visualization in openCV
        tmp1=np.concatenate((tmp,mask_normalized),axis=1)
        writer2.write(tmp1)
        mask_pre=mask

    writer.release()
    writer2.release()
    cv2.destroyAllWindows()
    end=time.time()
    print('Total time collapsed: ',end-start)
