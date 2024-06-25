# SUNIndx: Visual identification of sun stressed-areas of the fruit using a reflectance index (Sun Index) from hyperspectral images to predict sunscald-risk postharvest on ‘Granny Smith’ apples

## Software description
SUNIndx is a group of functions coded in Python to obtain a chlorophyll/carotenoid ratio from hyperspectral images of ‘Granny Smith’ apples at harvest, use it to identify sun stress regions on the fruit surface, and predict sunscald development postharvest [1]. The function and methodology described here can be manipulated and tailored to detect sun stress in different commodities based on hyperspectral images.
SUNIndx was created using Python 3.10.13 running in a Spyder 5 environment. This app used hyperspectral information to create an index based on chlorophyll a (reflectance values at 430 and 662 nm) and carotenoids (454 and 549 nm) bands. Later, this index is used to identify sun stress areas over the fruit and perform a sunscald-risk assessment. The App comprises two main functions, `Sun Stress Calibration` and `Sunscald Risk Prediction`, and four auxiliary functions: `void`, `gammaCorrection`, `plot_px`, and `fruit_edge`.

## Getting Started guide
### Step 1: install Spyder and Python interpreter
1-	New users are encouraged to install Python and Spyder from anaconda distribution at https://www.anaconda.com/download#download.
### Step 2: copy SUNIndx code
1-	Clone the project on your personal computer using git, or download a copy from https://github.com/renemogollon/SUNINDEX/tree/main 
### Step 3: install required packages
Required package for SUNIndx:
-	Pandas
-	Numpy
-	Cv2
-	Matplotlib
-	Seaborn
-	Os
  
This package can be installed by running the command pip install <package_name> or using Anaconda Navigator (https://docs.anaconda.com/anacondaorg/user-guide/packages/installing-packages/)  
### Step 4: Download datasets
Two datasets are provided in this repository: 
- Sun_Strees_Calibration_Data (https://github.com/renemogollon/SUNINDEX/tree/main/Sun_Stress_Calibration_Data)
  It contains RGB pictures of fruit sun-exposed sides with three different levels of sun stress: Clean (no stress), Mild, and Moderate. Spectral data from those three pictures is included in the CSV file in the folder.
- Sun_risk_Prediction_Data (https://github.com/renemogollon/SUNINDEX/tree/main/Sun_risk_Prediction_Data)
  It contains 12 RGB pictures from both fruit sides (non-exposed and sun-exposed) with three different levels of sun stress: Clean (no stress), Mild, and Moderate.
  Spectral data from those three pictures is included in the CSV file in the folder.

#### Dataset format:
`file`: RGB file name.\
`sunburn`: sun stress category:  Clean (no stress), Mild, or Moderate. For more information see [2]\
`side`: fruit side: nonexposed (NE), exposed (E). For more information see [1]\
`mask`: fruit number identification on the tray.\
`Npx`: pixel number identification on each fruit.\
`x`, `y`: x and y coordinates for each pixel per fruit.\
`Inc_aph`: sunscald incidences after storage. For more information see [1]\
`399` to `1000`: Reflectance spectral percentage between 399 and 1000 nm. More information about spectral pre-processing can be found in [1]\
`Ind_1`: ratio between chlorophyll (430, 662 nm) and carotenoid (454, 549 nm) wavelengths.  For more information see [1]

### Step 5: run SUNIndx:
1-	Open and run SUNIndx.py (https://github.com/renemogollon/SUNINDEX/blob/main/SUNIndx.py)

## Calibration example:
Sun Stress Calibration function creates a graphical user interface (GUI) using OpenCV as dependence and void, gammaCorrection, and plot_px as auxiliary functions. 
The parameters are sun index values by default for sun stress severities (colors): severe (black=1.16), moderate (red=1.38), mild (yellow=1.55); low severity (green) is set up by default as index values higher than the mild threshold (>1.55).
The OpenCV dependence creates a SUNIndx control panel pop-up window, which displays three trackbars. Each trackbar has an index range from 0 to 3 and executes the void function. A while loop creates and updates the images selected for calibration. These images are refreshed each time the user moves any of the trackbars.
Inside the while loop, there are two main processes. The first one is file image reading, where the two images are loaded and used as parameters of the gammaCorrection function. The second process consists of locating and identifying by color each of the pixels in each fruit in the image using as reference values of the previously calculated sun index and each of the thresholds of the trackbars; this process is carried out by plot_px function. Once the user is satisfied with the sun stress identification areas, he needs to press the ESC key to exit the function, close all pop-up windows, and print the threshold values on the console.

``` python
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
            
            plot_px(pic_1.replace('\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_1,spectral_data) 
            plot_px(pic_2.replace('\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_2,spectral_data)
            plot_px(pic_3.replace('\\Sun_Stress_Calibration_Data\\',''),color_black,color_red,color_yellow,img_3,spectral_data)
                    
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
            img_1 = cv.imread("\\Sun_Risk_Prediction_Data\\"+pic_1,1)
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


spectral_data_calibration = pd.read_csv("\\Sun_Stress_Calibration_Data\\Sun_Stress_Calibration_Data.csv", )

pic_1 = "\\Sun_Stress_Calibration_Data\\1.Clean_1 to 18_E.png"
pic_2 = "\\Sun_Stress_Calibration_Data\\1.Mild_127 to 144_E.png"
pic_3 = "\\Sun_Stress_Calibration_Data\\1.Moderate_1 to 18_E.png"

Sun_Stress_Calibration(spectral_data_calibration)
```
## Sunscald Risk Prediction example:
Sun Risk Prediction quantifies each pixel category based on each sun stress category [1]. Parameters’ functions are hyperspectral data (same data structure used by the Sun Stress Calibration function) and index threshold for each category (default severe=0.91, moderate=1.26, and mild=1.48).
It defines conditions based on the input thresholds to categorize the data into risk levels. These conditions are used to create a new column named 'P_Sun_Index' in the DataFrame. An empty data frame, which is used to save the categorized data, is initialized. This data frame is populated with information from each image to process, such as file name, number, and location (X and Y coordinate) of fruit on each image. Using this information, the function iterates over each unique file (image) and fruit on it to calculate the percentage of each sun stress category [1] (rejected, high-risk, low-risk), using the total number of pixels by fruit to calculate the percentage of pixels on each sun stress category/color (severe = grey, moderate = red, mild = yellow, and low = green). This function returns a data frame with the sun stress category for each fruit by image.
```python
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import os as os

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

   #### Main ######
spectral_data_risk_prediction = pd.read_csv("\\Sun_Risk_Prediction_Data\\Sun_Risk_Prediction_Data.csv", )
spectral_data_risk_prediction['slope']=spectral_data_risk_prediction['600']-spectral_data_risk_prediction['501']
spectral_data_risk_prediction[spectral_data_risk_prediction['slope']<0.31].file.unique()
noise = find_edge(spectral_data_risk_prediction[spectral_data_risk_prediction['slope']<0.31].file.unique()[0],spectral_data_risk_prediction,value=0.31)
  spectral_data_risk_prediction =  spectral_data_risk_prediction[spectral_data_risk_prediction['slope']>noise].copy()

prediction_final = Sun_Risk_Prediction(spectral_data_risk_prediction, reject=0.91,red=1.26,yellow=1.48)
prediction_final.to_csv('Sun_Risk_predition.csv',index=False)
```

## References
[1]	R. Mogollón, M. Mendoza, L. León, D. Rudell, C. A. Torres, Excluding sunscald from long-term storage of ‘Granny Smith’ apples, Postharvest Biology and Technology, 216, 2024, doi: 10.1016/j.postharvbio.2024.113044 \
[2]	O. Hernandez, C. A. Torres, M. A. Moya-León, M. C. Opazo, and I. Razmilic, Roles of the ascorbate-glutathione cycle, pigments and phenolics in postharvest ‘sunscald’ development on ‘granny smith’ apples (Malus domestica Borkh.), Postharvest Biol Technol, vol. 87, pp. 79–87, 2014, doi: 10.1016/j.postharvbio.2013.08.003 

