import os, cv2, random
import numpy as np
import scipy
import scipy.misc
import openslide
import pandas as pd
import lxml.etree as ET
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
from skimage import exposure
from xml_to_mask import xml_to_mask, get_num_classes, write_minmax_to_xml

################################################################################
################################ main function #################################
################################################################################

def get_wsi_patch(filename, patch_size=256, downsample=4, augment=0):
    '''
    takes a wsi and returns a random patch of patch_size
    downsample = >1 downsample of patch
    augment = [0,1] the percent of data that will be augmented

    '''
    try: augment = augment.numpy()
    except: augment = augment
    try: patch_size = patch_size.numpy()
    except: patch_size = patch_size
    try: filename = filename.numpy()
    except: filename = filename
    try: downsample = downsample.numpy()
    except: downsample = downsample

    wsi = openslide.OpenSlide(filename)

    l_dims = wsi.level_dimensions
    level = wsi.get_best_level_for_downsample(downsample + 0.1)

    # get or create wsi mask
    try:
        mask_path = '{}_MASK.png'.format(filename.decode().split('.')[0])
        xml_path = '{}.xml'.format(filename.decode().split('.')[0])
    except:
        mask_path = '{}_MASK.png'.format(filename.split('.')[0])
        xml_path = '{}.xml'.format(filename.split('.')[0])

    if not os.path.isfile(mask_path):
        save_wsi_thumbnail_mask(filename)
    mask = np.array(Image.open(mask_path))

    # test for xml and choose random annotated class for patch
    if os.path.isfile(xml_path):
        class_num = get_num_classes(xml_path)
        class_num = int(round(np.random.uniform(low=0, high=class_num-1)))
    else:
        class_num = 0

    print(class_num)
    region = get_patch(wsi, xml_path, class_num, l_dims, level, mask, patch_size, filename, downsample, augment)


    # region = get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, augment)

    # region = np.transpose(region, (2,0,1)) # [CWH]
    return region

################################################################################
################################# subfunctions #################################
################################################################################

def get_patch(wsi, xml_path, annotationID, l_dims, level, mask, patch_size, filename, downsample, augment):

    if augment > 0:
        # pad for affine
        patch_size = patch_size+4

    while True:

        if level == -1: # if no resolution works return white region
            print('{} broken | using white patch...'.format(filename))
            return np.ones((patch_size, patch_size,3))*255

        level_dims = l_dims[level]
        level_downsample = wsi.level_downsamples[level]
        thumbnail_size = float(max(mask.shape))
        mask_scale = thumbnail_size/max(level_dims)
        scale_factor = int(round(downsample / level_downsample))
        patch_width = patch_size*scale_factor

        try:

            if annotationID == 0:
                # track locations and vectorize mask
                [y_ind, x_ind] = np.indices(np.shape(mask))
                y_ind = y_ind.ravel()
                x_ind = x_ind.ravel()
                mask_vec = mask.ravel()

                # select random pixel with tissue
                idx = random.choice(np.argwhere(mask_vec==255))[0]
                x_mask = x_ind[idx]
                y_mask = y_ind[idx]

                # calc wsi patch start indicies
                x_start = int( (x_mask / mask_scale) - (patch_width/2))*level_downsample
                y_start = int( (y_mask / mask_scale) - (patch_width/2))*level_downsample

            else:
                # parse xml and get root
                tree = ET.parse(xml_path)
                root = tree.getroot()

                write_minmax_to_xml(xml_path, tree)

                locations = []
                # find all regions in annotation
                for Vert in root.findall("./Annotation[@Id='{}']/*/*/Vertices".format(annotationID)):
                    # get minmax points
                    Xmin = np.int32(Vert.attrib['Xmin'])
                    Ymin = np.int32(Vert.attrib['Ymin'])
                    Xmax = np.int32(Vert.attrib['Xmax'])
                    Ymax = np.int32(Vert.attrib['Ymax'])
                    locations.append([Xmin,Xmax,Ymin,Ymax])

                location = random.choice(locations)
                # find point in a random annotation
                x_start = int(round(np.random.uniform(low=location[0], high=location[1]) - (patch_width/2)))
                y_start = int(round(np.random.uniform(low=location[2], high=location[3]) - (patch_width/2)))

            region = wsi.read_region((int(x_start),int(y_start)), level, (patch_width,patch_width))

            if scale_factor > 1:
                region = region.resize((patch_size, patch_size), resample=1)
            region = np.array(region)[:,:,:3]

            if np.random.random() < augment:
                # augment image
                augment_patch(region)

            if augment > 0:
                # unpad
                region = region[2:-2,2:-2,:]

            return region

        except:
            print('{} broken for level {}'.format(filename,level))
            level -= 1
            print('\ttrying level {}'.format(level))

def get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, augment):
    return get_patch(wsi, None, 0, l_dims, level, mask, patch_size, filename, downsample, augment)

def save_wsi_thumbnail_mask(filename):
    '''
    saves a low resolution png mask of the tissue location in a WSI

    '''

    try: filename = filename.numpy()
    except: filename = filename
    wsi = openslide.OpenSlide(filename)

    def find_tissue_mask():
        thumbnail = wsi.get_thumbnail((thumbnail_size,thumbnail_size))
        thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10)))
        ret2,mask = cv2.threshold(thumbnail_blurred[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask[mask==0] = 1
        mask[mask==255] = 0
        return mask

    l_dims = wsi.level_dimensions
    thumbnail_size = min(2000., max(l_dims[0]))

    mask_path = '{}_MASK.png'.format(filename.split('.')[0])

    if not os.path.isfile(mask_path):
        print(filename)
        mask = find_tissue_mask()*255
        mask_PIL = Image.fromarray(mask)
        mask_PIL.save(mask_path)

def augment_patch(region):

    def colorshift(img, hbound=0.025, lbound=0.015): #Shift Hue of HSV space and Lightness of LAB space
        hShift=np.random.normal(0,hbound)
        lShift=np.random.normal(1,lbound)
        img=rgb2hsv(img)
        img[:,:,0]=(img[:,:,0]+hShift)
        img=hsv2rgb(img)
        img=rgb2lab(img)
        img[:,:,0]=exposure.adjust_gamma(img[:,:,0],lShift)
        img=lab2rgb(img)
        return img

    def PiecewiseAffine(img, points=8):
        ### piecwise affine ###
        rows, cols = img.shape[0], img.shape[1]
        src_cols = np.linspace(0, cols, points)
        src_rows = np.linspace(0, rows, points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        # add offset
        dst_rows = np.zeros(src[:, 1].shape) + src[:, 1]
        for i in list(range(points))[1:-1]:
            dst_rows[i::points] += np.random.normal(loc=0, scale=rows/(points*10), size=dst_rows[i::points].shape)
        dst_cols = np.zeros(src[:, 0].shape) + src[:, 0]
        dst_cols[points:-points] += np.random.normal(loc=0,scale=rows/(points*10), size=dst_cols[points:-points].shape)
        dst = np.vstack([dst_cols, dst_rows]).T
        # compute transform
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        # apply transform
        img = warp(img, tform, output_shape=(rows, cols))
        return img

    region = (region/255.).astype(np.float64)
    region = colorshift(region)
    region = PiecewiseAffine(region)
    region = np.uint8(region*255.)

    return region


def get_slide_label(filename, data_label_xlsx):
    data_label_xlsx = str(data_label_xlsx.numpy())
    filename = str(filename.numpy())
    # get slide label
    df = pd.read_excel(data_label_xlsx)
    name = filename.split('/')[-1]
    index = df.index[df['wsi']==name].tolist()
    if index == []:
        label = np.array([-1])
    else:
        label = np.array(df['class'][index])
    return label
