# Fetal_cartiotocography_extract_code

This is the code to extract the Fetal heart rate and Uterine contraction data from Fetal cardiotocography stored as images.
This page is used as a reference for the paper 'Machine learning model for classifying the result of fetal cardiotocography conducted in high-risk pregnancy'.
The method introduced in this paper proceeds in the following order.
  
  
### 1. FHR(UC) box detection through Hough transform algorithm  
```
v_phl = phl(fhr_box, threshold=100, line_length=100, theta=v_theta)     ## vertical lines of boxes
h_phl = phl(fhr_box, threshold=200, line_length=200, theta=h_theta)     ## horizontal lines of the fhr_box
h_phl_uc = phl(uc_box, threshold=200, line_length=200, theta=h_theta)   ## horizontal lines of the uc_box
```  
![fetal3](https://user-images.githubusercontent.com/67408403/155272543-612bd0c9-1eca-40c6-ab03-ff69af406802.png) 
  
### 2. Extracted FHR box example
![fetal1](https://user-images.githubusercontent.com/67408403/155272451-afc2264a-b1bc-43ca-ab0f-f77bcbaa186e.png)  
   
### 3. Waveform detection through pixel-wise deviation
```
def filtering_by_std(image, axis=0, threshold=25, upper=True, minimum_pix=500):
    image = np.std(image, axis=axis)
    if upper: image = image>threshold
    elif not upper: image = image<threshold
    image = rso(image, minimum_pix)
    return image
``` 
![fetal2](https://user-images.githubusercontent.com/67408403/155272503-0d9de5b6-4ba2-41dc-8f93-a13b6bfde801.png)  
   
### 4. Value scaling from pixel y position to measured heart rate values  
```
def scaler(data, h_pix, h_size, baseline=0, cut=False):
    HR = []
    for i in data:
        if np.isnan(i):
            HR.append(None)
            continue
        value = ((h_pix-int(i))/h_pix*h_size) + baseline
        if cut is True:
            value = round(value, 4)
        HR.append(value)
    return HR
```  
![fetal4](https://user-images.githubusercontent.com/67408403/155272569-7c959f11-49a8-46b2-b27e-51505ca6ddc5.png)  
