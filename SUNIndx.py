# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:04:08 2024

@author: rene.mogollon
"""

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import os as os

#### Sun Stress Calibration Fuction ######

def Sun_Stress_Calibration (spectral_data,black=11600,red=13800,yellow=15500):
    cv.namedWindow('control')
    cv.createTrackbar('black','control',black,30000, nothig)
    cv.createTrackbar('red','control',red,30000, nothig)
    cv.createTrackbar('yellow','control',yellow,30000, nothig)
    
    flag_color_black = -1
    flag_color_red = -1
    flag_color_yellow = -1
    
    
    while (1):
        color_black = cv.getTrackbarPos('black', 'control')/10000
        color_red = cv.getTrackbarPos('red', 'control')/10000
        color_yellow = cv.getTrackbarPos('yellow', 'control')/10000
        
        
        if ((flag_color_black != color_black) | (flag_color_red != color_red) | (flag_color_yellow != color_yellow)) :
                   
            img_1 = cv.imread(pic_1,1)
            img_1 = gammaCorrection(img_1, 2)
            img_1_copy = img_1.copy()
            img_2 = cv.imread(pic_2,1)
            img_2 = gammaCorrection(img_2, 2)
            img_2_copy = img_2.copy()
            img_3 = cv.imread(pic_3,1)
            img_3 = gammaCorrection(img_3, 2)
            img_3_copy = img_3.copy()
            
            plot_px(pic_1.replace('E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_1,spectral_data) 
            plot_px(pic_2.replace('E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_2,spectral_data)
            plot_px(pic_3.replace('E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_3,spectral_data)
                    
            cv.imshow(pic_1, img_1)
            cv.imshow(pic_2, img_2)
            cv.imshow(pic_3, img_3)
            cv.imshow('org. '+ pic_1, img_1_copy)
            cv.imshow('org. '+pic_2, img_2_copy)
            cv.imshow('org. '+pic_3, img_3_copy)
            
    
            flag_color_black = color_black
            flag_color_red = color_red
            flag_color_yellow = color_yellow
            
            
            
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break  
    
    plt.close()
    cv.destroyAllWindows()
    return(color_black,color_red,color_yellow)


#### Sun Risk Prediction Fuction ######


def Sun_Risk_Prediction (data,reject=0.91,red=1.26,yellow=1.48):
    'percentage of cluster according to its risk level'
    'input index 1 limits, categories (P1) cat_reject=6% cat_high_red=3% cat_high_yellow=55% '
    In_reject = reject
    In_red = red
    In_yellow = yellow
 
    conditions=[(data['Ind_1'] <=In_reject ),
                (data['Ind_1'] > In_reject) & (data['Ind_1'] <= In_red),
                (data['Ind_1'] > In_red) & (data['Ind_1'] <= In_yellow),
                (data['Ind_1'] > In_yellow)]
    values=['Reject','High','Medium','Low']
    data.loc[:,'P_In1']=np.select(conditions,values)
    risk_out=pd.DataFrame()
    for HSI_file in data['file'].unique():
        risk_out=risk_out.append(pd.DataFrame({'file':np.repeat(HSI_file,len(data[data['file']==HSI_file]['mask'].unique())),
                           'fruit':data[data['file']==HSI_file]['mask'].unique(),
                           'x':-1,'y':-1,'Px':-1,'P1Low':-1,'P1Medium':-1,'P1High':-1,'P1Reject':-1,                       
                           'P1':-1}),
                            ignore_index=True)
    for HSI_file in risk_out.file.unique():
        for fruit in risk_out[risk_out['file']==HSI_file]['fruit'].unique():
            print(HSI_file+' '+str(fruit))
            data_aux = data[(data['file']==HSI_file)& (data['mask']==fruit)]
            risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'Px']=len (data_aux)
            risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'x']=int(data_aux['x'].mean())
            risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'y']=int(data_aux['y'].mean())
            try:
                data_aux.groupby('P_In1').size()['Low']
            except:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Low']=0
            else:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Low']=data_aux.groupby('P_In1').size()['Low']
            try:
                data_aux.groupby('P_In1').size()['Medium']
            except:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Medium']=0
            else:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Medium']=data_aux.groupby('P_In1').size()['Medium']
            try:
                data_aux.groupby('P_In1').size()['High']
            except:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1High']=0
            else:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1High']=data_aux.groupby('P_In1').size()['High']
            try:
                data_aux.groupby('P_In1').size()['Reject']
            except:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Reject']=0
            else:
                risk_out.loc[(risk_out['fruit']==fruit) & (risk_out['file']==HSI_file),'P1Reject']=data_aux.groupby('P_In1').size()['Reject']
    risk_out['P1Low'] = risk_out['P1Low']*100/risk_out['Px']
    risk_out['P1Medium'] = risk_out['P1Medium']*100/risk_out['Px']
    risk_out['P1High'] = risk_out['P1High']*100/risk_out['Px']
    risk_out['P1Reject'] = risk_out['P1Reject']*100/risk_out['Px']
    ## P1 categories
    cat_reject=3
    cat_high_red=6
    cat_high_yellow=50
    cat_high_green=50

    for case in np.arange(0,len(risk_out)):
        if risk_out.P1Reject[case]>=cat_reject:
            risk_out.loc[case,'P1'] = 'Reject' 
            continue
        if risk_out.P1High[case]>=cat_high_red:
            risk_out.loc[case,'P1'] = 'High' 
            continue
        if risk_out.P1Medium[case]>=cat_high_yellow:
            risk_out.loc[case,'P1'] = 'Lowexp' 
            continue
        else:
            risk_out.loc[case,'P1'] = 'Low'
    
    

    predic_final = pd.DataFrame({'file':np.repeat(['1.Clean_1 to 18','1.Clean_19 to 36','1.Mild_1 to 18', '1.Mild_19 to 36','1.Moderate_1 to 18','1.Moderate_19 to 36'],18),
                                 'fruit':np.tile(np.arange(1,19),len(['1.Clean_1 to 18','1.Clean_19 to 36','1.Mild_1 to 18', '1.Mild_19 to 36','1.Moderate_1 to 18','1.Moderate_19 to 36'])),
                                 'back':-1,
                                 'front':-1,
                                 'final':-1})
    
    for tray in   ['1.Clean_1 to 18','1.Clean_19 to 36','1.Mild_1 to 18', '1.Mild_19 to 36','1.Moderate_1 to 18','1.Moderate_19 to 36']:
        print(tray)
        for fruit in np.arange(1,19):
            fruit
            predic_final.loc[(predic_final['file']==tray) & (predic_final['fruit']==fruit),'front'] = risk_out[(risk_out['file']==tray+'_E.png')&(risk_out['fruit']==fruit)]['P1'].values
            predic_final.loc[(predic_final['file']==tray) & (predic_final['fruit']==fruit),'back'] = risk_out[(risk_out['file']==tray+'_NE.png')&(risk_out['fruit']==fruit)]['P1'].values
    predic_final.loc[:,'final'] = predic_final['back']+predic_final['front']
    for case in predic_final.index:
        if 'Reject' in predic_final.loc[case,'final']:
           predic_final.loc[case,'final'] ="Reject"
           continue
        if 'High' in predic_final.loc[case,'final']:
            predic_final.loc[case,'final']='High'
            continue
        if 'Lowexp' in predic_final.loc[case,'final']:
            predic_final.loc[case,'final']="Lowexp"
            continue
        if 'Low' in predic_final.loc[case,'final']:
            predic_final.loc[case,'final']="Low"
            continue   
        if predic_final.loc[case,'final'] == 'VlowVlow':
            predic_final.loc[case,'final']="Vlow"
            continue
        else:
            predic_final.loc[case,'final']='Check'
            
    return (predic_final)

    


#### Axiliary Fucntions ######

def nothig (x):
    return

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv.LUT(src, table)

def plot_px(file,black,red,yellow,img,data):
    'file: HSI string to plot'
    'black, red, yellow, blue : limits'
    'img: cv file (.png)'
    'data: spectral dataframe '
    data_aux = data[data['file']==file]
    pixels=data_aux[data_aux['Ind_1']<black]
    for pxl in np.arange(0,len(pixels)):
          cv.circle(img,(int(pixels.iloc[pxl]['x']),int(pixels.iloc[pxl]['y'])),
                      radius=2, color= (180,180,180),thickness=-1)
    pixels=data_aux[(data_aux['Ind_1']>black) & (data_aux['Ind_1']<red)]
    for pxl in np.arange(0,len(pixels)):
          cv.circle(img,(int(pixels.iloc[pxl]['x']),int(pixels.iloc[pxl]['y'])),
                      radius=2, color= (0,17,255),thickness=-1)          
    pixels=data_aux[(data_aux['Ind_1']>red) & (data_aux['Ind_1']<yellow)]
    for pxl in np.arange(0,len(pixels)):
          cv.circle(img,(int(pixels.iloc[pxl]['x']),int(pixels.iloc[pxl]['y'])),
                      radius=2, color= (0,255,255),thickness=-1)
    pixels=data_aux[(data_aux['Ind_1']>yellow)]
    for pxl in np.arange(0,len(pixels)):
          cv.circle(img,(int(pixels.iloc[pxl]['x']),int(pixels.iloc[pxl]['y'])),
                      radius=2, color= (0,255,0),thickness=-1)

def find_edge (pic_1,data,value=0.31):
    cv.namedWindow('control')
    cv.createTrackbar('edge', 'control', int(value*10000), 10000, nothig)
    
    flag_edge = -1
    while (1):
        color_edge = cv.getTrackbarPos('edge', 'control')/10000
        if flag_edge != color_edge:
            img_1 = cv.imread("E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Risk_Prediction_Data\\"+pic_1,1)
            img_1 = gammaCorrection(img_1, 2)
            data_aux = data[data['file']==pic_1]
            pixels=data_aux[data_aux['slope']<color_edge]
            for pxl in np.arange(0,len(pixels)):
                  cv.circle(img_1,(int(pixels.iloc[pxl]['x']),int(pixels.iloc[pxl]['y'])),
                              radius=2, color= (255,0,255),thickness=-1)
            cv.imshow(pic_1, img_1)
            plt.close('all')
            flag_edge= color_edge
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break  
    print('slope value '+ str(color_edge))
    cv.destroyAllWindows()
    plt.close('all')
    return color_edge

#### Main ######


spectral_data_calibration = pd.read_csv("E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\Sun_Stress_Calibration_Data.csv", )

pic_1 = "E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\1.Clean_1 to 18_E.png"
pic_2 = "E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\1.Mild_127 to 144_E.png"
pic_3 = "E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Stress_Calibration_Data\\1.Moderate_1 to 18_E.png"

Sun_Stress_Calibration(spectral_data_calibration)


spectral_data_risk_prediction = pd.read_csv("E:\\rene.mogollon\\Documents\\manuscript under prep\\spectral indexes\\submission\\Github\\Sun_Risk_Prediction_Data\\Sun_Risk_Prediction_Data.csv", )

spectral_data_risk_prediction['slope']=spectral_data_risk_prediction['600']-spectral_data_risk_prediction['501']
spectral_data_risk_prediction[spectral_data_risk_prediction['slope']<0.31].file.unique()

noise = find_edge(spectral_data_risk_prediction[spectral_data_risk_prediction['slope']<0.31].file.unique()[0],spectral_data_risk_prediction,value=0.31)
   
spectral_data_risk_prediction =  spectral_data_risk_prediction[spectral_data_risk_prediction['slope']>noise].copy()

prediction_final = Sun_Risk_Prediction(spectral_data_risk_prediction, reject=0.91,red=1.26,yellow=1.48)
prediction_final.to_csv('Sun_Risk_predition.csv',index=False)
