import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects as rso
from skimage.transform import probabilistic_hough_line as phl

## image check function
def simple_show(img, figsize = (10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
    plt.close()

## function for filtering wave pixels from background pixels using standard deviation value of RGB values
def filtering_by_std(image, axis=0, threshold=25, upper=True, minimum_pix=500):
    image = np.std(image, axis=axis)
    if upper: image = image>threshold
    elif not upper: image = image<threshold
    image = rso(image, minimum_pix)
    return image

## function for abstracting y_values of pixels placed in the same place on the x-axis
def abstracter(image,x_range=0,x_screen=0,y_screen=0):
    HR_raw = []
    x_ = np.arange(0,x_range,1)
    for x in x_:
        y_idx = []
        for y in np.where(image[:,x]==1)[0]:
            if all([x<x_screen, y<y_screen]): continue
            y_idx.append(y)
        if len(y_idx) > 100:
            y_idx = []
        if y_idx:
            mean = np.mean(y_idx)
        else:
            mean = np.nan
        HR_raw.append(mean)
    return HR_raw

## function for scaling y-value of pixel to measurement value
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

## file path
base_path = pathlib.WindowsPath('''''')

## indicate FHR_boundary and UC_boundary roughly
## xl: left side of the box
## xr: right sidr of the box
## fhr_yt: top of the fhr box
## fhr_yb: bottom of the fhr box
## uc_yt: top of the uc box
## uc_yb: bottom of the uc box
xl, xr, fhr_yt, fhr_yb = 55, 775, 165, 390
uc_yt, uc_yb = 415, 550
length_list = []
pid_list = set()

## get pids for extraction
for i in base_path.glob('*.png'):
    name = i.stem.split('(')[0]
    pid_list.add(name)

## iterations for each pid
for p in pid_list:
    l = []
    l_uc = []
    for i in base_path.glob(f'{p}*.png'):
        print(i.stem.split('(')[-1][:-1])
        series = int(i.stem.split('(')[-1][:-1])
        color = imread(i).                              ## color image of the box. needed for waveform data extraction
        gray = rgb2gray(color)                          ## gray image of the box. needed for boundary detection
        # simple_show(gray)
        
        reversed_gray = 1 - gray
        # simple_show(reversed_gray)

        r_gray = np.ones(reversed_gray.shape)
        # simple_show(r_gray)

        r_gray[reversed_gray < 0.01] = 0
        fhr_box = r_gray[fhr_yt:fhr_yb, xl:xr]      
        uc_box = r_gray[uc_yt:uc_yb, xl:xr]
        # simple_show(fhr_box)
        # simple_show(uc_box)

        angle = [0, np.pi / 2]                          ## angles of the lines which is for detected hough transform algorithm
        v_theta = np.array(angle[:-1], dtype='double')
        h_theta = np.array(angle[1:], dtype='double')   

        v_phl = phl(fhr_box, threshold=100, line_length=100, theta=v_theta)     ## vertical lines of boxes
        h_phl = phl(fhr_box, threshold=200, line_length=200, theta=h_theta)     ## horizontal lines of the fhr_box
        h_phl_uc = phl(uc_box, threshold=200, line_length=200, theta=h_theta)   ## horizontal lines of the uc_box

        x_values = [x[0][0] for x in v_phl]
        y_values = [y[0][1] for y in h_phl]
        y_values_uc = [y_uc[0][1] for y_uc in h_phl_uc]

        xl_new = xl + np.array(x_values).min()
        xr_new = xl + np.array(x_values).max()
        fhr_yt_new = fhr_yt + np.array(y_values).min()
        fhr_yb_new = fhr_yt + np.array(y_values).max()
        uc_yt_new = uc_yt + np.array(y_values_uc).min()
        uc_yb_new = uc_yt + np.array(y_values_uc).max()

        length = xr_new-xl_new
        length_list.append(length)
        # simple_show(r_gray[fhr_yt_new:fhr_yb_new, xl_new:xr_new], figsize=(10,5))     ## new fhr_box
        # simple_show(r_gray[uc_yt_new:uc_yb_new, xl_new:xr_new], figsize=(10,5))       ## new uc_box

        fhr_box_color = color[fhr_yt_new:fhr_yb_new, xl_new:xr_new]
        uc_box_color = color[uc_yt_new:uc_yb_new, xl_new:xr_new]
        # simple_show(color[fhr_yt_new:fhr_yb_new, xl_new:xr_new], figsize=(10,5))      ## new fhr_box color
        # simple_show(color[uc_yt_new:uc_yb_new, xl_new:xr_new], figsize=(10,5))        ## new uc_box color

        fhr_std = filtering_by_std(fhr_box_color, axis=2, threshold=15, upper=True, minimum_pix=50)     ## get waveform filtered by standard deviation of RGB pixels
        fhr_green = np.sum(fhr_box_color, axis=2) < 450                                                 ## get waveform filtered by sum of pixels
        # simple_show(fhr_std, figsize=(10, 5))
        # simple_show(fhr_green, figsize=(10, 5))

        HR_raw = np.all([fhr_std, fhr_green], axis=0)
        # simple_show(HR_raw, figsize=(10, 5))

        HR_raw = abstracter(HR_raw, x_range=avg_length-1, x_screen=0, y_screen=0)                       ## get y value of fhr by meaning y values filtered
        HR = scaler(HR_raw, fhr_yb_new-fhr_yt_new, 210, baseline=30, cut=False)                         ## transform y value to measured heart rate value
        # plt.figure(figsize=(10,5))
        # plt.xlim([0, avg_length])
        # plt.ylim([30, 240])
        # plt.plot(HR)
        # plt.show()
        # plt.close()
        l.append((series, HR))

        uc_black = np.min(uc_box_color, axis=2) < 100
        # simple_show(uc_black, figsize=(10, 5))
        
        UC_raw = abstracter(uc_black, x_range=avg_length-1, x_screen=0, y_screen=0) #142ppi
        UC = scaler(UC_raw, uc_yb_new-uc_yt_new, 100, baseline=0, cut=False) #142ppi
        # plt.figure(figsize=(10,5))
        # plt.xlim([0, avg_length])
        # plt.ylim([0, 100])
        # plt.plot(UC)
        # plt.show()
        # plt.close()
        l_uc.append((series, UC))
        
    l.sort(key=lambda x: x[0])
    l_uc.sort(key=lambda x: x[0])
    fhr = []
    for f in l:
        fhr.extend(f[1])
        print(len(fhr))
    # plt.figure(figsize=(25,3))
    # plt.xlim([0, avg_length*len(l)])
    # plt.ylim([30,240])
    # plt.plot(fhr)
    # plt.show()
    # plt.close()

    uc = []
    for u in l_uc:
        uc.extend(u[1])
    # plt.figure(figsize=(25,3))
    # plt.xlim([0,avg_length*len(l_uc)])
    # plt.ylim([0, 100])
    # plt.plot(uc)
    # plt.show()
    # plt.close()

    # print(l)
    # avg_length = np.array(length_list).mean()
    # print(int(avg_length))

    data = pd.DataFrame(zip(fhr, uc), columns=['FHR','UC'])
    # data.plot(figsize=(25,3))
    # plt.show()

    # data.to_csv('''''')
