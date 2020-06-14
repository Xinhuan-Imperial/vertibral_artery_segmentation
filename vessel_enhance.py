#!/usr/bin/python
# -*- coding:utf-8 -*-
import itk,time,math,os,pydicom,numpy as np
import SimpleITK as sitk

def stack_to_volume(filedir,num_slices,Venc,crop_window):
    file_list=os.listdir(filedir)
    name_list=[]
    for i in file_list[0:num_slices]:
        tmp0=os.path.splitext(i)[0]
        tmp1=os.path.splitext(i)[1]
        if tmp1=='.img' and tmp0[0:3]=='ana':
            name_list.append(i)
    data_array=np.zeros([crop_window[3]-crop_window[1],crop_window[2]-crop_window[0],len(name_list)],dtype=np.float)
    for i,name in enumerate(name_list):
        tmp1=os.path.join(filedir,name)
        anatomic_image=sitk.ReadImage(tmp1,sitk.sitkFloat32)
        anatomic=sitk.GetArrayFromImage(anatomic_image)
        velocity_name=tmp1.replace('anatomic','velocity')
        velocity_image=sitk.ReadImage(velocity_name,sitk.sitkFloat32)
        velocity=sitk.GetArrayFromImage(velocity_image)
        angle=velocity/Venc*math.pi
        magnitude=2*anatomic*np.sin(angle)   
        mask=velocity>0
        magnitude=np.array(magnitude.squeeze())*mask.squeeze()
        data_array[:,:,i]=magnitude[crop_window[1]:crop_window[3]:1,crop_window[0]:crop_window[2]:1]
    return data_array  

# calculate the vesselness using Heissian matrix. Sigma is 1,3,6,9. The calculated vesselness is the maximum of the four response at different sigma values
def calc_vesselness(input_image_array,voxel_size):
    input_image=itk.GetImageFromArray(input_image_array)
    sigma=1/math.pow(voxel_size[0]*voxel_size[1]*voxel_size[2],1/3)
    heissian_image=itk.hessian_recursive_gaussian_image_filter(input_image,sigma=sigma)
    vesselness_filter=itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype('float')].New()
    vesselness_filter.SetInput(heissian_image)
    vesselness_filter.SetAlpha1(0.5)
    vesselness_filter.SetAlpha2(0.5)
    vessel_array=100*itk.GetArrayFromImage(vesselness_filter)
    for sigma in [3,6,9]:
        heissian_image=itk.hessian_recursive_gaussian_image_filter(input_image,sigma=sigma)
        vesselness_filter.SetInput(heissian_image)
        data1=itk.GetArrayFromImage(vesselness_filter)
        vessel_array=np.maximum(vessel_array,100*data1)
    return vessel_array

if __name__=='__main__':
    start=time.time()
    input_folder='E:/sub2/artery/co2'
    crop_window=np.array([126,179,370,307],dtype=np.int16)#x and y coordinate of the left upper corner followed by the right lower corner, e.g., [126 179 370 307]
    #CO2:120,resting:80
    Venc=120
    voxel_size=[0.3906,0.3906,0.3906]
    num_slices=600#we will process a volume with num_slices slices each time for efficiency
    stacked_image=stack_to_volume(input_folder,num_slices,Venc,crop_window)  
    vesselness=calc_vesselness(stacked_image,voxel_size) 
    enhanced_vesselness=vesselness*stacked_image
    sitk.WriteImage(sitk.GetImageFromArray(enhanced_vesselness),'./segmented_vessel.nii.gz')

    #otsu_filter = sitk.OtsuThresholdImageFilter()
    #otsu_filter.SetInsideValue(0)
    #otsu_filter.SetOutsideValue(1)
    #enhanced_vesselness_img=sitk.GetImageFromArray(enhanced_vesselness)
    #seg = otsu_filter.Execute(enhanced_vesselness_img)
    #sitk.WriteImage(seg,'./segmented_vessel.nii.gz')
    end=time.time()
    print('Total time collapsed: ',end-start)