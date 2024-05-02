## SUNIndx: Visual identification of sun stressed-areas of the fruit using a reflectance index (Sun Index) from hyperspectral images to predict sunscald-risk postharvest on ‘Granny Smith’ apples

### Illustrative examples 
---
To use SUNIndx, the user needs a Python interpreter and the dependencies. Data sets used in this example are also available on the repository (different spectral image datasets can be available upon request). 
Spectral image datasets are presented in CSV format, contain fruit identification tags and locations, extracted with Superannotate software (v1.1.0, Sunnyvale, CA, USA), and pre-processed by pixel. 
Detail on spectral prepossessing can be found in Mogollon et al. (2024, in press).
The user needs to open the SUNIndx.py file and the data available on Sun__Stress_Calibration_Data and Sun_Risk_Prediction_Data
Sun Stress Calibration function is called, using the default threshold values for sun stress identification as parameters. Three pop-up windows are displayed, one called ‘control panel’, which contains the trackbars where the user can manipulate the sun stress thresholds, and one for each calibration image. To end the calibration process and obtain the sun stress threshold on the console, the ESC key should be pressed.
The Sunscald Risk Prediction function can be used independently or after the calibration process is done. This function captures the file location of the dataset of interest, with the same data structure as the spectral calibration dataset (CSV file), and the sun stress thresholds (default severe=0.91, moderate=1.26, and mild=1.48). after this function quantifies the percentage of sun-stressed pixels from each severity (Rejected, High-risk, Low-risk), predictions of both fruit sides are compared to consolidate sunscald risk assessment, and the most extreme is selected and saved in a CSV file in the working directory.
