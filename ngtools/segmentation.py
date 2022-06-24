# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:52:32 2021

@author: gabrielemilioherreraoropeza
"""

#############################################
#                 Imports                   #
#############################################

import os, json, random, itertools
import warnings

warnings.filterwarnings("ignore")

from os import listdir
from os.path import isfile, join
from collections import Counter
from math import pi, ceil
import czifile as zis
import numpy as np
import xmltodict
from cellpose import models, plot
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.filters import (threshold_otsu, threshold_isodata, threshold_li, threshold_mean,
                                 threshold_minimum, threshold_triangle, threshold_yen, threshold_sauvola,
                                 gaussian, threshold_multiotsu)
from skimage import exposure, img_as_float
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import cv2
import pandas as pd
from photutils.detection import find_peaks
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from spatialentropy import leibovici_entropy
from sklearn.cluster import KMeans
#from deepcell.applications import NuclearSegmentation
from scipy import stats

#############################################
#     Functions & Classes | Segmentation    #
#############################################


def get_array_czi(
        filename,
        replacevalue=False,
        remove_HDim=True,
        return_addmd=False
        ):
    """
    Get the pixel data of the CZI file as multidimensional NumPy.Array
    :param filename: filename of the CZI file
    :param replacevalue: replace arrays entries with a specific value with Nan
    :param remove_HDim: remove the H-Dimension (Airy Scan Detectors)
    :param return_addmd: read the additional metadata
    :return: cziarray - dictionary with the dimensions and its positions
    :return: metadata - dictionary with CZI metadata
    :return: additional_metadata_czi - dictionary with additional CZI metadata
    """

    # get metadata
    metadata = get_metadata_czi(filename)

    # get additional metadata
    additional_metadata_czi = get_additional_metadata_czi(filename)

    # get CZI object and read array
    czi = zis.CziFile(filename)
    cziarray = czi.asarray()

    # check for H dimension and remove
    try:
        if remove_HDim and metadata['Axes'][0] == 'H':
            metadata['Axes'] = metadata['Axes'][1:]
            cziarray = np.squeeze(cziarray, axis=0)
    except:
        pass

    # get additional information about dimension order etc.
    try:
        dim_dict, dim_list, numvalid_dims = get_dimorder(metadata['Axes'])
        metadata['DimOrder CZI'] = dim_dict
    except:
        metadata['DimOrder CZI'] = "unknown"

    try:
        if cziarray.shape[-1] == 3:
            pass
        else:
            cziarray = np.squeeze(cziarray, axis=len(metadata['Axes']) - 1)
    except:
        pass

    if replacevalue:
        cziarray = replace_value(cziarray, value=0)

    # close czi file
    czi.close()

    return cziarray, metadata, additional_metadata_czi


def get_metadata_czi(filename, dim2none=False):
    """
    Returns a dictionary with CZI metadata.
    Information CZI Dimension Characters:
    '0': 'Sample',  # e.g. RGBA
    'X': 'Width',
    'Y': 'Height',
    'C': 'Channel',
    'Z': 'Slice',  # depth
    'T': 'Time',
    'R': 'Rotation',
    'S': 'Scene',  # contiguous regions of interest in a mosaic image
    'I': 'Illumination',  # direction
    'B': 'Block',  # acquisition
    'M': 'Mosaic',  # index of tile for compositing a scene
    'H': 'Phase',  # e.g. Airy detector fibers
    'V': 'View',  # e.g. for SPIM
    :param filename: filename of the CZI image
    :param dim2none: option to set non-existing dimension to None
    :return: metadata - dictionary with the relevant CZI metainformation
    """

    # get CZI object and read array
    czi = zis.CziFile(filename)
    #mdczi = czi.metadata()

    # parse the XML into a dictionary
    metadata = create_metadata_dict()
    try:
        metadatadict_czi = xmltodict.parse(czi.metadata())
    except:
        print("WARNING! Metadata could not be read!")
        return metadata

    # get directory and filename etc.
    try:
        metadata['Directory'] = os.path.dirname(filename)
    except:
        metadata['Directory'] = 'Unknown'
    try:
        metadata['Filename'] = os.path.basename(filename)
    except:
        metadata['Filename'] = 'Unknown'
    metadata['Extension'] = 'czi'
    metadata['ImageType'] = 'czi'

    # add axes and shape information
    metadata['Axes'] = czi.axes
    metadata['Shape'] = czi.shape

    # determine pixel type for CZI array
    metadata['NumPy.dtype'] = str(czi.dtype)

    # check if the CZI image is an RGB image depending on the last dimension entry of axes
    if czi.axes[-1] == 3:
        metadata['isRGB'] = True

    metadata['Information'] = metadatadict_czi['ImageDocument']['Metadata']['Information']
    try:
        metadata['PixelType'] = metadata['Information']['Image']['PixelType']
    except KeyError as e:
        print('Key not found:', e)
        metadata['PixelType'] = None

    metadata['SizeX'] = np.int(metadata['Information']['Image']['SizeX'])
    metadata['SizeY'] = np.int(metadata['Information']['Image']['SizeY'])

    try:
        metadata['SizeZ'] = np.int(metadata['Information']['Image']['SizeZ'])
    except:
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = np.int(metadata['Information']['Image']['SizeC'])
    except:
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    channels = []
    for ch in range(metadata['SizeC']):
        try:
            channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                            ['Channels']['Channel'][ch]['ShortName'])
        except:
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                ['Channels']['Channel']['ShortName'])
            except:
                channels.append(str(ch))

    metadata['Channels'] = channels

    try:
        metadata['SizeT'] = np.int(metadata['Information']['Image']['SizeT'])
    except:
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = np.int(metadata['Information']['Image']['SizeM'])
    except:
        if dim2none:
            metadata['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = np.int(metadata['Information']['Image']['SizeB'])
    except:

        if dim2none:
            metadata['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = np.int(metadata['Information']['Image']['SizeS'])
    except:
        if dim2none:
            metadata['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['Scaling'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']
        metadata['XScale'] = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['YScale'] = float(metadata['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        metadata['XScale'] = np.round(metadata['XScale'], 3)
        metadata['YScale'] = np.round(metadata['YScale'], 3)
        try:
            metadata['XScaleUnit'] = metadata['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
            metadata['YScaleUnit'] = metadata['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
        except:
            metadata['XScaleUnit'] = None
            metadata['YScaleUnit'] = None
        try:
            metadata['ZScale'] = float(metadata['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            metadata['ZScale'] = np.round(metadata['ZScale'], 3)
            try:
                metadata['ZScaleUnit'] = metadata['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
            except:
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
        except:
            if dim2none:
                metadata['ZScale'] = metadata['XScaleUnit']
            if not dim2none:
                # set to isotropic scaling if it was single plane only
                metadata['ZScale'] = metadata['XScale']
    except:
        metadata['Scaling'] = None

    # try to get software version
    try:
        metadata['SW-Name'] = metadata['Information']['Application']['Name']
        metadata['SW-Version'] = metadata['Information']['Application']['Version']
    except KeyError as e:
        print('Key not found:', e)
        metadata['SW-Name'] = None
        metadata['SW-Version'] = None

    try:
        metadata['AcqDate'] = metadata['Information']['Image']['AcquisitionDateAndTime']
    except KeyError as e:
        print('Key not found:', e)
        metadata['AcqDate'] = None

    try:
        metadata['Instrument'] = metadata['Information']['Instrument']
    except KeyError as e:
        print('Key not found:', e)
        metadata['Instrument'] = None

    if metadata['Instrument'] is not None:

        # get objective data
        try:
            metadata['ObjName'] = metadata['Instrument']['Objectives']['Objective']['@Name']
        except:
            metadata['ObjName'] = None

        try:
            metadata['ObjImmersion'] = metadata['Instrument']['Objectives']['Objective']['Immersion']
        except:
            metadata['ObjImmersion'] = None

        try:
            metadata['ObjNA'] = np.float(metadata['Instrument']['Objectives']['Objective']['LensNA'])
        except:
            metadata['ObjNA'] = None

        try:
            metadata['ObjID'] = metadata['Instrument']['Objectives']['Objective']['@Id']
        except:
            metadata['ObjID'] = None

        try:
            metadata['TubelensMag'] = np.float(metadata['Instrument']['TubeLenses']['TubeLens']['Magnification'])
        except:
            metadata['TubelensMag'] = None

        try:
            metadata['ObjNominalMag'] = np.float(metadata['Instrument']['Objectives']['Objective']['NominalMagnification'])
        except KeyError as e:
            print('Key not found:', e)
            metadata['ObjNominalMag'] = None

        try:
            metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
        except:
            metadata['ObjMag'] = None

        # get detector information
        try:
            metadata['DetectorID'] = metadata['Instrument']['Detectors']['Detector']['@Id']
        except:
            metadata['DetectorID'] = None

        try:
            metadata['DetectorModel'] = metadata['Instrument']['Detectors']['Detector']['@Name']
        except:
            metadata['DetectorModel'] = None

        try:
            metadata['DetectorName'] = metadata['Instrument']['Detectors']['Detector']['Manufacturer']['Model']
        except:
            metadata['DetectorName'] = None

        # delete some key from dict
        del metadata['Instrument']

    # check for well information

    metadata['Well_ArrayNames'] = []
    metadata['Well_Indices'] = []
    metadata['Well_PositionNames'] = []
    metadata['Well_ColId'] = []
    metadata['Well_RowId'] = []
    metadata['WellCounter'] = None

    try:
        allscenes = metadata['Information']['Image']['Dimensions']['S']['Scenes']['Scene']
        for s in range(metadata['SizeS']):
            well = allscenes[s]
            metadata['Well_ArrayNames'].append(well['ArrayName'])
            metadata['Well_Indices'].append(well['@Index'])
            metadata['Well_PositionNames'].append(well['@Name'])
            metadata['Well_ColId'].append(well['Shape']['ColumnIndex'])
            metadata['Well_RowId'].append(well['Shape']['RowIndex'])

        # count the content of the list, e.g. how many time a certain well was detected
        metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])
        # count the number of different wells
        metadata['NumWells'] = len(metadata['WellCounter'].keys())

    except:
        print('Key not found: S')
        print('No Scence or Well Information detected:')


    # for getting binning

    try:
        channels = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
        for channel in range(len(channels)):
            cuch = channels[channel]
            metadata['Binning'].append(cuch['DetectorSettings']['Binning'])

    except KeyError as e:
        print('Key not found:', e)
        print('No Binning Found')

    del metadata['Information']
    del metadata['Scaling']

    # close CZI file
    czi.close()

    return metadata


def get_additional_metadata_czi(filename):
    """
    Returns a dictionary with additional CZI metadata.
    :param filename: filename of the CZI image
    :return: additional_czimd - dictionary with the relevant OME-TIFF metainformation
    """

    # get CZI object and read array
    czi = zis.CziFile(filename)

    # parse the XML into a dictionary
    additional_czimd = {}
    try:
        metadatadict_czi = xmltodict.parse(czi.metadata())
    except:
        print("WARNING! Additional metadata could not be read.")
        return additional_czimd

    try:
        additional_czimd['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except:
        additional_czimd['Experiment'] = None

    try:
        additional_czimd['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except:
        additional_czimd['HardwareSetting'] = None

    try:
        additional_czimd['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except:
        additional_czimd['CustomAttributes'] = None

    try:
        additional_czimd['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['DisplaySetting'] = None

    try:
        additional_czimd['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['Layers'] = None

    # close CZI file
    czi.close()

    return additional_czimd


def create_metadata_dict():
    """
    A Python dictionary will be created to hold the relevant metadata.
    :return: metadata - dictionary with keys for the relevant metadata
    """

    metadata = {'Directory': None,
                'Filename': None,
                'Extension': None,
                'ImageType': None,
                'Name': None,
                'AcqDate': None,
                'TotalSeries': None,
                'SizeX': None,
                'SizeY': None,
                'SizeZ': None,
                'SizeC': None,
                'SizeT': None,
                'Sizes BF': None,
                'DimOrder BF': None,
                'DimOrder BF Array': None,
                'DimOrder CZI': None,
                'Axes': None,
                'Shape': None,
                'isRGB': None,
                'ObjNA': None,
                'ObjMag': None,
                'ObjID': None,
                'ObjName': None,
                'ObjImmersion': None,
                'XScale': None,
                'YScale': None,
                'ZScale': None,
                'XScaleUnit': None,
                'YScaleUnit': None,
                'ZScaleUnit': None,
                'DetectorModel': [],
                'DetectorName': [],
                'DetectorID': None,
                'InstrumentID': None,
                'Channels': [],
                'ImageIDs': [],
                'NumPy.dtype': None,
                'Binning': []
                }

    return metadata


def replace_value(data, value=0):
    """
    Replace specifc values in array with NaN
    :param data: NumPy.Array
    :param value: value inside array to be replaced with Nan
    :return: data - array with new values
    """

    data = data.astype('float')
    data[data == value] = np.nan

    return data


def get_dimorder(dimstring):
    """
    Get the order of dimensions from dimension string
    :param dimstring: string containing the dimensions
    :return: dims_dict - dictionary with the dimensions and its positions
    :return: dimindex_list - list with indices of dimensions
    :return: numvalid_dims - number of valid dimensions
    """

    dimindex_list = []
    dims = ['B', 'S', 'T', 'C', 'Z', 'Y', 'X', '0']
    dims_dict = {}

    for d in dims:

        dims_dict[d] = dimstring.find(d)
        dimindex_list.append(dimstring.find(d))

    numvalid_dims = sum(i > 0 for i in dimindex_list)

    return dims_dict, dimindex_list, numvalid_dims


def check_channels(lst_of_lsts):
    """
    Checks that the channels for every image are the same

    Parameters
    ----------
    lst_of_lsts : list
        List containing a list of the channels for each image.

    Returns
    -------
    test : bool
        True if all channels for all images are the same.
        False if not.

    """
    test = True
    for n in range(len(lst_of_lsts)-1):
        if Counter(lst_of_lsts[n]) != Counter(lst_of_lsts[n+1]):
            test = False
    return test


def wk_array(array, axes):
    """
    Converts the raw array into a proper format to perform the analysis.

    Parameters
    ----------
    array : array
        Array of the image.
    axes : string
        Order of the axes.

    Returns
    -------
    working_array : array
        Array in the proper format to perform the analysis.

    """

    if axes == 'SCYX0':
        working_array = array[0]

    elif axes == 'BCYX0':
        working_array = array[0]

    elif axes == 'CYX0':
        working_array = array

    elif axes == 'YX' or axes == 'XY':
        working_array = array

    #elif axes == 'BVCTZYX0':
    #    working_array = array[0,0,:,0,0,:,:]

    #elif axes == 'VBTCZYX0':
    #    working_array = array[0,0,0,:,13,:,:]

    return working_array


def _cellpose(image, diameter = None, GPU = None):
    """
    Run cellpose for nuclear segmentation

    Parameters
    ----------
    image : array
        Image to segment as array.
    diameter : integer, optional
        Aproximate average diameter. The default is None.
    GPU : bool, optional
        Use GPU if available. The default is None.

    Returns
    -------
    masks : array
        Masks obtained.
    flows : TYPE
        DESCRIPTION.

    """

    model = models.Cellpose(gpu = GPU, model_type = 'nuclei')
    channels = [0, 0]
    masks, flows, _, _ = model.eval(image, diameter = diameter, channels = channels, resample = True)

    return masks, flows


def _deepcell(image, scale):
    """
    Run DeepCell method for nuclear segmentation.

    Parameters
    ----------
    image : array
        Image to segment as array.
    scale : float
        micrometers per pixel.

    Returns
    -------
    mask : array
        Mask obtained.

    """
    new_arr = []
    ys = []
    for y in image:
        xs = []
        for x in y:
            xs.append([x])
        ys.append(xs)
    new_arr.append(ys)
    new_arr = np.array(new_arr)

    app = None #NuclearSegmentation()
    y_pred = app.predict(new_arr, image_mpp = scale)

    mask = []
    for y in y_pred[0]:
        xs = []
        for x in y:
            xs.append(x[0])
        mask.append(xs)
    mask = np.array(mask)

    return mask

def nucleus_layers_fast(image, mask, xscale):
    image = image.copy()
    mask = mask.copy()
    mask_internal = mask.copy()
    mask_external = mask.copy()
    bin_mask = mask.copy()
    bin_mask[bin_mask>0] = 1

    iter_1um = round(1 / float(xscale))  # Number of iterations equivalent to 1um
    kernel = np.ones((3, 3), np.uint8)

    eroded_1um = cv2.erode(bin_mask, kernel, iterations=iter_1um)
    mask_dilated_1um = cv2.dilate(mask_external, kernel, iterations=iter_1um)
    dilated_1um = cv2.dilate(bin_mask, kernel, iterations=iter_1um)

    eroded_1um = np.array(eroded_1um)
    dilated_1um = np.array(dilated_1um)

    internal = bin_mask - eroded_1um
    mask_internal[internal==0] = 0
    internal_props = regionprops(mask_internal, intensity_image=image)
    internal_avg_int = [ceil(internal_props[n]['intensity_mean']) for n in range(len(internal_props))]
    internal_total_int = [np.sum(internal_props[n]['image_intensity']) for n in range(len(internal_props))]

    external = dilated_1um - bin_mask
    mask_dilated_1um[external == 0] = 0
    external_props = regionprops(mask_dilated_1um, intensity_image=image)
    external_avg_int = [ceil(external_props[n]['intensity_mean']) for n in range(len(external_props))]
    external_total_int = [np.sum(external_props[n]['image_intensity']) for n in range(len(external_props))]

    # get core
    mask_props = regionprops(mask, intensity_image=image)
    total_int = [np.sum(mask_props[n]['image_intensity']) for n in range(len(mask_props))]
    area_0 = [ceil(mask_props[n]['area']) for n in range(len(mask_props))]

    core_avg_int = []
    core_total_int = []
    kernel = np.ones((3, 3), np.uint16)

    # ## testing, to speeed up identifiying core mask
    # to_erode = np.array([True]*len(area_0))
    # to_erode_index = np.array(list(range(1, len(area_0)+1)))
    # area_after = area_0.copy()
    # area_before = [0]*len(area_0)
    # core_mask = mask.copy()
    #
    # while any(to_erode):
    #     to_erode_mask = core_mask.copy()
    #     to_erode_mask[np.isin(to_erode_mask,to_erode_index[to_erode], invert=True)] = 0
    #     eroded_mask = cv2.erode(to_erode_mask, kernel, iterations=1)
    #
    #

    for cell in range(len(area_0)):
        bin_mask = np.zeros(image.shape)
        bin_mask[mask == (cell+1)] = 1
        bin_mask = np.uint16(bin_mask)
        a0 = area_0[cell]
        aN = a0
        prev_area_n = None
        while a0/ 2 < aN:
            if aN == prev_area_n:
                break
            bin_mask = cv2.erode(bin_mask, kernel, iterations=1)
            itereprops = regionprops(bin_mask, intensity_image=image)
            prev_area_n = aN
            aN = itereprops[0]['area']
        core_avg_int.append(ceil(itereprops[0]['intensity_mean']))
        core_total_int.append(np.sum(itereprops[0]['image_intensity']))


    #while any(np.divide(area_0, 2) < area_n):
    #    if area_n == prev_area_n:
    #        break
    #    iter_bin_mask = cv2.erode(iter_bin_mask, kernel, iterations = 1)
    #    itereprops = regionprops(iter_bin_mask, intensity_image = image)
    #    prev_area_n = area_n
    #    area_n = [ceil(itereprops[n]['area']) for n in range(len(itereprops))]

    #core = iter_bin_mask
    #core_props = regionprops(core, intensity_image=image)
    #core_avg_int = [ceil(core_props[n]['intensity_mean']) for n in range(len(core_props))]
    #core_total_int = [np.sum(core_props[n]['image_intensity']) for n in range(len(core_props))]




    return(total_int, core_avg_int, core_total_int, internal_avg_int, internal_total_int, external_avg_int, external_total_int)





def nucleus_layers(image, mask, cellID, xscale):
    """
    Generates masks corresponding to nucleus layers.

    Parameters
    ----------
    image : array
        Image of the nuclei as array.
    mask : array
        Masks of the nuclei as array.
    cellID : int
        Mask number of nucleus of interest.
    xscale : float
        Number of um equivalent to one pixel.

    Returns
    -------
    core : array
        Mask of the core of the nucleus of interest.
    internal : array
        Mask of the internal ring of the nucleus of interest.
    external : array
        Mask of the external ring of the nucleus of interest.

    """

    image = image.copy()
    mask = mask.copy()

    iter_1um = round(1/float(xscale)) # Number of iterations equivalent to 1um
    kernel = np.ones((3, 3), np.uint8)

    bin_mask = np.zeros(image.shape)
    bin_mask[mask == cellID] = 1

    eroded_1um = cv2.erode(bin_mask, kernel, iterations = iter_1um)
    dilated_1um = cv2.dilate(bin_mask, kernel, iterations = iter_1um)

    eroded_1um = np.array(eroded_1um)
    dilated_1um = np.array(dilated_1um)

    internal = bin_mask - eroded_1um
    external = dilated_1um - bin_mask

    bin_mask = np.uint16(bin_mask)

    props = regionprops(bin_mask, intensity_image = image)
    for p in props:
        if p['label'] == 1:
                area_0 = p['area']

    area_n = area_0
    kernel = np.ones((3, 3), np.uint16)

    iter_bin_mask = bin_mask.copy()
    prev_area_n = None

    while (area_0/2) < area_n:
        if area_n == prev_area_n:
            break
        iter_bin_mask = cv2.erode(iter_bin_mask, kernel, iterations = 1)
        props = regionprops(iter_bin_mask, intensity_image = image)
        prev_area_n = area_n
        for p in props:
            if p['label'] == 1:
                area_n = p['area']

    core = iter_bin_mask

    internal[internal == 1] = cellID
    core[core == 1] = cellID
    external[external == 1] = cellID

    internal = internal.astype(np.uint16)
    core = core.astype(np.uint16)
    external = external.astype(np.uint16)

    return core, internal, external


def find_avg_intensity(image, mask, cellID):
    """
    Finds average pixel intensity from an intensity image by using a mask.

    Parameters
    ----------
    image : array-like
        Array of intensity image .
    mask : array-like
        Array of mask.
    cellID : int
        Number of cell of interest.

    Returns
    -------
    avg_intensity : int
        Average pixel intensity.

    """
    image = np.array(image.copy())
    mask = np.array(mask.copy())
    avg_intensity = np.average(image[mask == cellID])

    try:
        avg_intensity = round(avg_intensity)
    except:
        avg_intensity = avg_intensity

    return avg_intensity


def find_sum_intensity(image, mask, cellID):
    """
    Calculates sum of pixel intensity from an intensity image by using a mask.

    Parameters
    ----------
    image : array-like
        Array of intensity image .
    mask : array-like
        Array of mask.
    cellID : int
        Number of cell of interest.

    Returns
    -------
    sum_intensity : int
        Sum of pixel intensity of the area covered by the mask.

    """
    image = np.array(image.copy())
    mask = np.array(mask.copy())
    sum_intensity = np.sum(image[mask == cellID])

    return sum_intensity


def get_threshold_img(image, thresh_option):
    """
    Generate threshold image

    Parameters
    ----------
    image : array-like image
        Intensity image.
    thresh_option : str
        Thresholding option.

    Returns
    -------
    thresh_img : array-like
        Threshold image.

    """
    ### --- Adaptive Otsu
    if thresh_option.lower() == 'adaptive_otsu':
        fltd = gaussian(image, 3)
        th = threshold_otsu(fltd)
        thresh_img = fltd > th

    ### --- Otsu
    elif thresh_option.lower() == 'otsu':
        th = threshold_otsu(image)
        thresh_img = image > th

    ### --- Isodata
    elif thresh_option.lower() == 'isodata':
        th = threshold_isodata(image)
        thresh_img = image > th

    ### --- Li
    elif thresh_option.lower() == 'li':
        th = threshold_li(image)
        thresh_img = image > th

    ### --- Mean
    elif thresh_option.lower() == 'mean':
        th = threshold_mean(image)
        thresh_img = image > th

    ### --- Minimum
    elif thresh_option.lower() == 'minimum':
        th = threshold_minimum(image)
        thresh_img = image > th

    ### --- Triangle
    elif thresh_option.lower() == 'triangle':
        th = threshold_triangle(image)
        thresh_img = image > th

    ### --- Yen
    elif thresh_option.lower() == 'yen':
        th = threshold_yen(image)
        thresh_img = image > th

    ### --- Sauvola
    elif thresh_option.lower() == 'sauvola':
        th = threshold_sauvola(image)
        thresh_img = image > th

    return thresh_img, round(th)


def binary_img(thresh_img):
    """
    Generates binary image and closes small holes.

    Parameters
    ----------
    thresh_img : Boolean array
        Output of thresholding.

    Returns
    -------
    bin_img : array-like
        Binary image.

    """
    bin_img = np.uint8(thresh_img)

    # Close small holes inside foreground objects
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    return bin_img


def gammaCorrection(img, gamma):
    """
    Reduces intensity range by correcting image gamma.

    Parameters
    ----------
    img : array-like
        Intensity image.
    gamma : float, optional
        Gamma Value. The default is gamma.

    Returns
    -------
    gamma_corrected : array_like
        gamma-corrected image.

    """
    image = img_as_float(img)
    gamma_corrected = exposure.adjust_gamma(image, gamma)

    return gamma_corrected


def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = normalize99(img[:,:,i])
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


class NuclearGame_Segmentation(object):

    def __init__(self, indir, outdir=None):
        """
        Start Nuclear Game.
        Parameters
        ----------
        indir : string
            Is the path to the folder where all the microscope images that will be analysed
            are found.

        Returns
        -------
        None.

        """

        self.path_read = indir

        ## F: append / to path
        if indir.endswith("/"):
            self.path_read = indir
            indir = indir.strip("/")
        else:
            self.path_read = indir + "/"


        # TODO: add compatibility to more formats.
        formats = [".czi",
                   #".tiff",
                   #".tif"
                   ]

        print("WARNING... In this version (v0.1.2) only CZI files are supported (Note: Z-Stack and time lapse are not currently supported)")

        # Generate dictionary that will contain all the generated data
        self.data = {}
        self.data["files"] = {}

        files = [f for f in listdir(self.path_read) if isfile(join(self.path_read, f))]

        # check if folder contain correct file formats
        ext_list = [os.path.splitext(f)[1] for f in files]
        if not any(f in ext_list for f in formats):
            raise ValueError("Ops... No valid format found in the given path!")

        # Creat out folder in the same path
        outdir = indir if outdir is None else join(outdir, os.path.basename(indir))
        self.path_save = join(outdir, 'out_ng/')
        if os.path.isdir(self.path_save):
            n = 1
            while os.path.isdir(self.path_save):
                self.path_save = join(outdir, f'out_ng ({n})/')
                n += 1
        os.makedirs(self.path_save, exist_ok=True)

    def get_file_name(self, _format = ".czi", getall = False):
        """
        Gets the file names in a given path.
        Parameters
        ----------
        _format : string
                Is the format of the files that will be analysed

        Returns
        -------
        None.

        """

        self.image_format = _format

        files = [f for f in listdir(self.path_read) if isfile(join(self.path_read, f)) and f.lower().endswith(self.image_format)]

        if getall == False:
            while True:
                no_files = input(f'\nAnalyse all ({len(files)}) {self.image_format} files or select one (all/one)? ')
                if no_files.lower() == "all" or no_files.lower() == "one":
                    break
                else:
                    print(f"The input {no_files} is not valid, try again...")


            if no_files.lower() == "one":
                print("\n")
                for file in files:
                    print(file)
                while True:
                    new_files = input("\nEnter name of file to analyse: ")
                    if new_files in files:
                        files = [new_files]
                        break
                    else:
                        print(f"The given file name '{new_files}' is not valid, try again...")

            print("\nFiles to be analysed: \n")
        for file in files:
            _file = file.replace(self.image_format, "")
            self.data["files"][_file] = {}
            self.data["files"][_file]["path"] = self.path_read + file  ##F: corrected path
            print(f"\t{_file}", f"(format: {self.image_format.upper()})")


    def read_files(self):
        """
        Reads every file, generates arrays, and obtains metadata

        Returns
        -------
        None.

        """

        if self.image_format == ".czi":
            for file in self.data["files"]:
                self.data["files"][file]["array"], self.data["files"][file]["metadata"], self.data["files"][file]["add_metadata"] = get_array_czi(filename = self.data["files"][file]["path"])

        # TODO: support TIFF files
        elif self.image_format == ".tiff" or self.image_format == ".tif":
            pass


    def identify_channels(self, channels = None, marker = None):
        """
        Assign a name to each channel

        Returns
        -------
        None.

        """

        if channels == None:
            if self.image_format == ".czi":
                if len(self.data["files"]) > 1:
                    lsts_ch = [self.data["files"][file]['metadata']['Channels'] for file in self.data["files"]]
                    test_ch = check_channels(lsts_ch)
                    if test_ch == False:
                        raise ValueError("Channels of files are different!")
                    else:
                        self.data["channels_info"] = {}
                        for n, channel in enumerate(lsts_ch[0]):
                            marker = input(f"Insert name of marker in channel {channel}: ")
                            self.data["channels_info"][marker] = n
                elif len(self.data["files"]) == 1:
                    self.data["channels_info"] = {}
                    for file in self.data["files"]:
                        for n, channel in enumerate(self.data["files"][file]['metadata']['Channels']):
                            self.data["channels_info"][input(f"Insert name of marker in channel {channel}: ")] = n


                while True:
                    self.data["dna_marker"] = input(f"\nWhich marker is the DNA marker (nuclear staining) ({'/'.join(self.data['channels_info'].keys())})? ")
                    if self.data["dna_marker"] in list(self.data['channels_info'].keys()):
                        break
                    else:
                        print(f"{self.data['dna_marker']} is not in the list of markers! Try again...")

            elif self.image_format == ".tiff" or self.image_format == ".tif":
                pass
        else:
            self.data["channels_info"] = {}
            for n, channel in enumerate(channels):
                self.data["channels_info"][channel] = n
                self.data["dna_marker"] = marker


    def nuclear_segmentation(self, method = "cellpose", diameter = None, gamma_corr = False, gamma = 0.25, dc_scaleCorr = None, GPU = False):
        """
        Perform nuclear segmentation.

        Parameters
        ----------
        diameter : Integer, optional
            Approximate nuclear diameter. The default is None.

        Returns
        -------
        None.

        """

        if method.lower() == "cellpose":
            self.seg_method = "cellpose"
            for n, file in enumerate(self.data["files"]):
                print(f"\nPerforming segmentation on file {n+1} of {len(self.data['files'])} \n")
                self.data["files"][file]['working_array'] = wk_array(self.data["files"][file]['array'],
                                                                     self.data["files"][file]['metadata']['Axes'])

                nuclei = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                if gamma_corr:
                    nuclei = gammaCorrection(img = nuclei, gamma = gamma)
                self.data["files"][file]["masks"], self.data["files"][file]["flows"] = _cellpose(nuclei,
                                                                                       diameter = diameter,
                                                                                       GPU = GPU)
        elif method.lower() == "deepcell":
            self.check_pxScale()
            self.seg_method = "deepcell"
            for file in tqdm(self.data["files"]):
                self.data["files"][file]['working_array'] = wk_array(self.data["files"][file]['array'],
                                                                     self.data["files"][file]['metadata']['Axes'])

                nuclei = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                if gamma_corr:
                    nuclei = gammaCorrection(img = nuclei, gamma = gamma)
                if dc_scaleCorr == None:
                    dc_scaleCorr = 1
                self.data["files"][file]["masks"] = _deepcell(image = nuclei,
                                                              scale = self.data["files"][file]['metadata']['XScale'] * dc_scaleCorr)



    def show_segmentation(self, file):
        """
        Shows nuclear segmentation.

        Returns
        -------
        fig : plot
            Image plot of nuclear segmentation.

        """

        image = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
        mask = self.data["files"][file]["masks"].copy()

        if self.seg_method == "cellpose":

            flows = self.data["files"][file]["flows"]
            channels = [0, 0]

            fig = plt.figure(figsize=(15,6))
            plot.show_segmentation(fig, image, mask, flows[0], channels = channels)

            plt.tight_layout()

        elif self.seg_method == "deepcell":
            y, x = image.shape
            temp = np.ma.zeros((y, x, 3), dtype = 'uint8')
            for cell in np.unique(mask):
                r = random.randint(1, 255)
                g = random.randint(1, 255)
                b = random.randint(1, 255)
                if cell != 0:
                    temp[mask == cell] = [r, g, b]
            masked = np.ma.masked_where(temp == 0, temp)
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
            img0 = image.copy()
            if img0.shape[0] < 4:
                img0 = np.transpose(img0, (1,2,0))
            if img0.shape[-1] < 3 or img0.ndim < 3:
                img0 = image_to_rgb(img0, channels=[0,0])
            else:
                if img0.max()<=50.0:
                    img0 = np.uint8(np.clip(img0*255, 0, 1))
            axes[0].imshow(img0)
            axes[0].axis('off')
            axes[1].imshow(image, 'gray')
            axes[1].imshow(masked, alpha = 0.8)
            axes[1].axis('off')
            plt.tight_layout()

        return fig


    def check_pxScale(self, xscale = None, yscale = None):
        """
        Checks whether x and y scales are provided or not. If not, it asks for one.

        Returns
        -------
        None.

        """
        xflag = True
        yflag = True

        for file in self.data["files"]:

            if self.data["files"][file]['metadata']['XScale'] == 0.0 or self.data["files"][file]['metadata']['XScale'] == None:
                if xscale == None:
                    while xflag:
                        try:
                            xscale = float(input("Enter upp for X axis (e.g. 0.227): "))
                            self.data["files"][file]['metadata']['XScale'] = xscale
                            xflag = False
                            break
                        except:
                            print("Invalid input! Try again...\n")
                self.data["files"][file]['metadata']['XScale'] = xscale

            if self.data["files"][file]['metadata']['YScale'] == 0.0 or self.data["files"][file]['metadata']['YScale'] == None:
                if yscale == None:
                    while yflag:
                        try:
                            yscale = float(input("Enter upp for Y axis (e.g. 0.227): "))
                            self.data["files"][file]['metadata']['YScale'] = yscale
                            yflag = False
                            break
                        except:
                            print("Invalid input! Try again...\n")
                self.data["files"][file]['metadata']['YScale'] = yscale


    def nuclear_features(self, xscale = None, yscale=None):
        """
        Measure first pool of nuclear features.

        Returns
        -------
        None.

        """

        self.check_pxScale(xscale = xscale, yscale = yscale)

        for file in tqdm(self.data["files"]):

            self.data["files"][file]["nuclear_features"] = self.generate_dict_nf()

            mask = self.data["files"][file]["masks"]
            image = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]]

            props = regionprops(mask, intensity_image = image)

            for p in props:
                if p['label'] > 0:

                    self.data["files"][file]["nuclear_features"]["cellID"].append(p['label'])

                    area = p['area'] * (self.data["files"][file]['metadata']['XScale'] * self.data["files"][file]['metadata']['YScale'])
                    self.data["files"][file]["nuclear_features"]["nuclear_area"].append(round(area))

                    self.data["files"][file]["nuclear_features"][f"avg_intensity_{self.data['dna_marker']}"].append(round(p['intensity_mean']))

                    perimeter = p['perimeter'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["nuclear_perimeter"].append(round(perimeter))

                    circularity = 4 * pi * (area / perimeter ** 2)
                    self.data["files"][file]["nuclear_features"]["circularity"].append(round(circularity, 3))

                    self.data["files"][file]["nuclear_features"]["eccentricity"].append(round(p['eccentricity'], 3))

                    self.data["files"][file]["nuclear_features"]["solidity"].append(round(p['solidity'], 3))

                    major_axis = p['axis_major_length'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["major_axis"].append(round(major_axis, 1))

                    minor_axis = p['axis_minor_length'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["minor_axis"].append(round(minor_axis, 1))

                    axes_ratio = minor_axis / major_axis
                    self.data["files"][file]["nuclear_features"]["axes_ratio"].append(round(axes_ratio, 3))

                    cY, cX = p['centroid']
                    self.data["files"][file]["nuclear_features"]["x_pos"].append(round(cX))
                    self.data["files"][file]["nuclear_features"]["y_pos"].append(round(cY))

                    self.data["files"][file]["nuclear_features"]["angle"].append(p["orientation"])

                    self.data["files"][file]["nuclear_features"]["imageID"].append(file)


    def generate_dict_nf(self):
        """
        Creates dictionary for the nuclear features

        Returns
        -------
        dct_df : dict
            Dictionary that will containd the values of the nuclear features.

        """

        dct_df = {
            'cellID': [],
            f'avg_intensity_{self.data["dna_marker"]}': [],
            'nuclear_area': [],
            'nuclear_perimeter': [],
            'major_axis': [],
            'minor_axis': [],
            'axes_ratio': [],
            'circularity': [],
            'eccentricity': [],
            'solidity': [],
            'x_pos': [],
            'y_pos': [],
            'angle': [],
            'imageID': []
            }

        return dct_df


    def plot_boxplot_hist(self, feature = "nuclear_area"):
        """
        Generates boxplot and histogram for a desired nuclear feature.

        Parameters
        ----------
        feature : string, optional
            Desired nuclear feature to show. The default is "nuclear_area".

        Returns
        -------
        fig : plot
            Boxplot-Histogram.

        """

        ft2show = [l for file in self.data["files"] for l in self.data["files"][file]["nuclear_features"][feature]]

        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)
        fig.add_trace(go.Box(x = ft2show, boxpoints = 'suspectedoutliers', fillcolor = 'rgba(7,40,89,0.5)',
                             line_color = 'rgb(7,40,89)', showlegend = False, name = ''), row = 1, col = 1)
        fig.add_trace(go.Histogram(x = ft2show, histnorm = 'probability', marker_color = 'rgb(7,40,89)',
                                  name = feature, showlegend = False), row = 2, col = 1)
        fig.update_layout(title = f"{feature} distribution",
                          xaxis = dict(autorange = True, showgrid = True, zeroline = True, gridwidth=1), width = 1000,
                          height = 400, template = "plotly_white")

        return fig


    def print_features(self):
        """
        Prints measured nuclear features.

        Returns
        -------
        None.

        """
        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                if ft != "cellID" and ft != "imageID":
                    print(ft)
            break


    def print_files(self):
        """
        Prints files.

        Returns
        -------
        None.

        """
        for file in self.data["files"]:
            print(file)


    def add_nf(self):
        """
        Generate a list containing additional nuclear features.

        Returns
        -------
        fts2add : list
            List of additional nuclear features.

        """
        fts2add = []

        for ch in self.data["channels_info"]:
            if ch == self.data["dna_marker"]:
                fts2add.append(f"avg_intensity_core_{ch}")
                fts2add.append(f"avg_intensity_internal_ring_{ch}")
                fts2add.append(f"avg_intensity_external_ring_{ch}")
                fts2add.append(f"total_intensity_core_{ch}")
                fts2add.append(f"total_intensity_internal_ring_{ch}")
                fts2add.append(f"total_intensity_external_ring_{ch}")
                fts2add.append(f"total_intensity_{ch}")
            else:
                fts2add.append(f"avg_intensity_{ch}")
                fts2add.append(f"total_intensity_{ch}")

        channels = [ch for ch in self.data["channels_info"] if ch != self.data["dna_marker"]]

        for subset in findsubsets(channels, 2):
            ch1, ch2 = subset
            fts2add.append(f"{ch1}_x_{ch2}")

        fts2add.append("_x_".join(channels))

        return fts2add


    def add_nf2file(self):
        """
        Add additional nuclear features to dictionaries of files.

        Returns
        -------
        None.

        """
        add_nf = self.add_nf()
        for file in self.data["files"]:
            for ft in add_nf:
                self.data["files"][file]["nuclear_features"][ft] = []


    def add_nuclear_features(self):
        """
        Measure additional nuclear features.

        Returns
        -------
        None.

        """
        self.add_nf2file()

        for file in tqdm(self.data["files"]):
            for ch in self.data["channels_info"]:
                mask = self.data["files"][file]["masks"]
                image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                if ch == self.data["dna_marker"]:
                    out_nuclear_layers = nucleus_layers_fast(image, mask,
                                                             xscale = self.data["files"][file]['metadata']['XScale'])
                    self.data["files"][file]["nuclear_features"][f"avg_intensity_core_{ch}"] = out_nuclear_layers[1]
                    self.data["files"][file]["nuclear_features"][f"avg_intensity_internal_ring_{ch}"] = out_nuclear_layers[3]
                    self.data["files"][file]["nuclear_features"][f"avg_intensity_external_ring_{ch}"] = out_nuclear_layers[5]
                    self.data["files"][file]["nuclear_features"][f"total_intensity_{ch}"] = out_nuclear_layers[0]
                    self.data["files"][file]["nuclear_features"][f"total_intensity_core_{ch}"] = out_nuclear_layers[2]
                    self.data["files"][file]["nuclear_features"][f"total_intensity_internal_ring_{ch}"] = out_nuclear_layers[4]
                    self.data["files"][file]["nuclear_features"][f"total_intensity_external_ring_{ch}"] = out_nuclear_layers[6]

                else:
                    cellprops = regionprops(mask, intensity_image=image)
                    avg_intensities = [ceil(cellprops[n]['intensity_mean']) for n in range(len(cellprops))]
                    total_intensties = [np.sum(cellprops[n]['image_intensity']) for n in range(len(cellprops))]
                    self.data["files"][file]["nuclear_features"][f"avg_intensity_{ch}"] = avg_intensities
                    self.data["files"][file]["nuclear_features"][f"total_intensity_{ch}"] = total_intensties

            channels = [ch for ch in self.data["channels_info"] if ch != self.data["dna_marker"]]

            for subset in findsubsets(channels, 2):
                ch1, ch2 = subset
                intensity1 = self.data["files"][file]["nuclear_features"][f"avg_intensity_{ch1}"]
                intensity2 = self.data["files"][file]["nuclear_features"][f"avg_intensity_{ch2}"]
                for i1, i2 in zip(intensity1, intensity2):
                    self.data["files"][file]["nuclear_features"][f"{ch1}_x_{ch2}"].append(i1 * i2)

            intensity1 = self.data["files"][file]["nuclear_features"][f"avg_intensity_{channels[0]}"]
            intensity2 = self.data["files"][file]["nuclear_features"][f"avg_intensity_{channels[1]}"]
            intensity3 = self.data["files"][file]["nuclear_features"][f"avg_intensity_{channels[2]}"]
            for i1, i2, i3 in zip(intensity1, intensity2, intensity3):
                self.data["files"][file]["nuclear_features"]["_x_".join(channels)].append(i1 * i2 * i3)


    def find_dna_peaks(self, box_size = 10, zoom_box_size = 300):
        """
        Finds number of DNA peaks

        Parameters
        ----------
        box_size : int, optional
            Side size (px) of box for finding high intensity dots. The default is 10.
        zoom_box_size : int, optional (enter None for using whole image)
            Side size of box (px) for accelerating the finding of the dots. The whole nucleus
            should fit inside of the box.

        Returns
        -------
        None.

        """

        for file in tqdm(self.data["files"]):

            self.data["files"][file]["nuclear_features"]["dna_peaks"] = []

            for cell in self.data["files"][file]["nuclear_features"]["cellID"]:

                _index = self.data["files"][file]["nuclear_features"]["cellID"].index(cell)

                masks = self.data["files"][file]["masks"].copy()
                masks[masks != cell] = 0
                masks[masks == cell] = 1

                nucleus = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                nucleus[masks != 1] = 0

                if zoom_box_size != None:
                    half_zoom_box = int(zoom_box_size / 2)
                    cY = int(self.data["files"][file]["nuclear_features"]["y_pos"][_index])
                    cX = int(self.data["files"][file]["nuclear_features"]["x_pos"][_index])
                    cY_low = cY - half_zoom_box
                    cY_high = cY + half_zoom_box
                    cX_low = cX - half_zoom_box
                    cX_high = cX + half_zoom_box
                    if (cY-half_zoom_box) < 0:
                        cY_low = 0
                    if (cY+half_zoom_box) > len(nucleus):
                        cY_high = len(nucleus)
                    if (cX-half_zoom_box) < 0:
                        cX_low = 0
                    if (cX+half_zoom_box) > len(nucleus[0]):
                        cX_high = len(nucleus[0])
                    nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
                    masks = masks[cY_low:cY_high, cX_low:cX_high]

                ignore_mask = np.zeros(masks.shape)
                ignore_mask[masks == 0] = True
                ignore_mask[masks != 0] = False
                ignore_mask = ignore_mask.astype(bool)

                try:
                    bkg = Background2D(nucleus, 3, mask = ignore_mask)

                    th = detect_threshold(data = nucleus, nsigma = 0, mask_value = 0, background = bkg.background)

                    peak_tb = find_peaks(data = nucleus, threshold = th, mask = ignore_mask, box_size = box_size)

                    try:
                        peak_df = peak_tb.to_pandas()
                        lst_remove = []

                        for index, row in peak_df.iterrows():
                            x_up = row["x_peak"] + 5
                            x_down = row["x_peak"] - 5
                            y_up = row["y_peak"] + 5
                            y_down = row["y_peak"] -5
                            temp_df = peak_df[((peak_df["x_peak"] > x_down) & (peak_df["x_peak"] < x_up)) & ((peak_df["y_peak"] > y_down) & (peak_df["y_peak"] < y_up))]
                            if len(temp_df) > 1:
                                sorted_df = temp_df.sort_values(by = "peak_value", ascending = False)
                                flag = True
                                for index2, row2 in sorted_df.iterrows():
                                    if flag == True:
                                        flag = False
                                        pass
                                    elif flag == False:
                                        lst_remove.append(index2)

                        peak_df_fltd = peak_df.drop(lst_remove)

                        no = len(peak_df_fltd)
                        self.data["files"][file]["nuclear_features"]["dna_peaks"].append(no)

                    except:
                        self.data["files"][file]["nuclear_features"]["dna_peaks"].append(0)

                except:
                    self.data["files"][file]["nuclear_features"]["dna_peaks"].append(np.nan)


    def find_dna_dots(self, zoom_box_size = 300):
        """
        Finds number of DNA dots

        Parameters
        ----------
        zoom_box_size : int, optional (enter None for using whole image)
            Side size of box (px) for accelerating the finding of the dots. The whole nucleus
            should fit inside of the box.

        Returns
        -------
        None.

        """

        for file in tqdm(self.data["files"]):

            self.data["files"][file]["nuclear_features"]["dna_dots"] = []
            self.data["files"][file]["nuclear_features"]["dna_dots_size_median"] = []

            for cell in self.data["files"][file]["nuclear_features"]["cellID"]:

                _index = self.data["files"][file]["nuclear_features"]["cellID"].index(cell)

                masks = self.data["files"][file]["masks"].copy()
                masks[masks != cell] = 0
                masks[masks == cell] = 1

                nucleus = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                nucleus[masks != 1] = 0

                if zoom_box_size != None:
                    half_zoom_box = int(zoom_box_size / 2)
                    cY = int(self.data["files"][file]["nuclear_features"]["y_pos"][_index])
                    cX = int(self.data["files"][file]["nuclear_features"]["x_pos"][_index])
                    cY_low = cY - half_zoom_box
                    cY_high = cY + half_zoom_box
                    cX_low = cX - half_zoom_box
                    cX_high = cX + half_zoom_box
                    if (cY-half_zoom_box) < 0:
                        cY_low = 0
                    if (cY+half_zoom_box) > len(nucleus):
                        cY_high = len(nucleus)
                    if (cX-half_zoom_box) < 0:
                        cX_low = 0
                    if (cX+half_zoom_box) > len(nucleus[0]):
                        cX_high = len(nucleus[0])
                    nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
                    masks = masks[cY_low:cY_high, cX_low:cX_high]

                ignore_mask = np.zeros(masks.shape)
                ignore_mask[masks == 0] = True
                ignore_mask[masks != 0] = False
                ignore_mask = ignore_mask.astype(bool)

                try:
                    bkg = Background2D(nucleus, 3, mask = ignore_mask)

                    th = detect_threshold(data = nucleus, nsigma = 0.5, mask_value = 0, background = bkg.background)

                    sigma = 3.0 * gaussian_fwhm_to_sigma

                    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
                    kernel.normalize()
                    segm = detect_sources(data = nucleus, threshold = th, npixels = 5, kernel = kernel)

                    if segm is None:
                        self.data["files"][file]["nuclear_features"]["dna_dots"].append(0)
                        self.data["files"][file]["nuclear_features"]["dna_dots_size_median"].append(0)
                    else:
                        cat = SourceCatalog(nucleus, segm)
                        columns = ['label', 'xcentroid', 'ycentroid', 'area']
                        dots_df = cat.to_table(columns).to_pandas()
                        n_dots = len(dots_df)
                        self.data["files"][file]["nuclear_features"]["dna_dots"].append(n_dots)
                        median_size_dots = dots_df["area"].median() * (self.data["files"][file]['metadata']['XScale'] * self.data["files"][file]['metadata']['YScale'])
                        self.data["files"][file]["nuclear_features"]["dna_dots_size_median"].append(median_size_dots)
                except:
                    self.data["files"][file]["nuclear_features"]["dna_dots_size_median"].append(np.nan)
                    self.data["files"][file]["nuclear_features"]["dna_dots"].append(np.nan)


    def spatial_entropy(self, d = 5, zoom_box_size = 300):
        """
        Finds spatial entropy for each nucleus.

        Parameters
        ----------
        d : int, optional
            Side size (px) of box for finding co-occurrences. The default is 5.
        zoom_box_size : int, optional (enter None for using whole image)
            Side size of box (px) for accelerating the finding of the dots. The whole nucleus
            should fit inside of the box.

        Returns
        -------
        None.

        """

        for file in tqdm(self.data["files"]):

            self.data["files"][file]["nuclear_features"]["spatial_entropy"] = []

            for cell in self.data["files"][file]["nuclear_features"]["cellID"]:

                _index = self.data["files"][file]["nuclear_features"]["cellID"].index(cell)

                masks = self.data["files"][file]["masks"].copy()
                masks[masks != cell] = 0
                masks[masks == cell] = 1

                nucleus = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                nucleus[masks != 1] = 0

                if zoom_box_size != None:
                    half_zoom_box = int(zoom_box_size / 2)
                    cY = int(self.data["files"][file]["nuclear_features"]["y_pos"][_index])
                    cX = int(self.data["files"][file]["nuclear_features"]["x_pos"][_index])
                    cY_low = cY - half_zoom_box
                    cY_high = cY + half_zoom_box
                    cX_low = cX - half_zoom_box
                    cX_high = cX + half_zoom_box
                    if (cY-half_zoom_box) < 0:
                        cY_low = 0
                    if (cY+half_zoom_box) > len(nucleus):
                        cY_high = len(nucleus)
                    if (cX-half_zoom_box) < 0:
                        cX_low = 0
                    if (cX+half_zoom_box) > len(nucleus[0]):
                        cX_high = len(nucleus[0])
                    nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
                    masks = masks[cY_low:cY_high, cX_low:cX_high]

                pp = np.array([[nx, ny] for ny in range(len(nucleus)) for nx in range(len(nucleus[ny])) if nucleus[ny][nx] != 0])

                lst_types = []

                p10, p20, p30, p40, p50, p60, p70, p80, p90 = np.percentile(nucleus[nucleus != 0], [10, 20, 30, 40, 50, 60, 70, 80, 90])

                for l in pp:
                    x, y = l
                    if nucleus[y][x] < p10:
                        lst_types.append("1")
                    elif nucleus[y][x] >= p10 and nucleus[y][x] < p20:
                        lst_types.append("2")
                    elif nucleus[y][x] >= p20 and nucleus[y][x] < p30:
                        lst_types.append("3")
                    elif nucleus[y][x] >= p30 and nucleus[y][x] < p40:
                        lst_types.append("4")
                    elif nucleus[y][x] >= p40 and nucleus[y][x] < p50:
                        lst_types.append("5")
                    elif nucleus[y][x] >= p50 and nucleus[y][x] < p60:
                        lst_types.append("6")
                    elif nucleus[y][x] >= p60 and nucleus[y][x] < p70:
                        lst_types.append("7")
                    elif nucleus[y][x] >= p70 and nucleus[y][x] < p80:
                        lst_types.append("8")
                    elif nucleus[y][x] >= p80 and nucleus[y][x] < p90:
                        lst_types.append("9")
                    else:
                        lst_types.append("10")

                types = np.array(lst_types)

                lb_ent = leibovici_entropy(pp, types, d)

                self.data["files"][file]["nuclear_features"]["spatial_entropy"].append(round(lb_ent.entropy, 3))



    def positive2marker(self, frac_covered = 0.8, thresh_method = "triangle"):
        """
        Identify whether a cell is positive to a nuclear marker.

        Parameters
        ----------
        frac_covered : float, optional
            Fraction of the nucleus covered by marker. The default is 0.8.
        thresh_option : str, optional
            Thresholding method. The default is "triangle".

        Returns
        -------
        None.

        """
        for ch in tqdm(self.data["channels_info"]):
            if ch == self.data["dna_marker"]:
                continue
            for file in self.data["files"]:

                image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                th_img, th = get_threshold_img(image, thresh_method)
                bin_img = binary_img(th_img)

                mask = self.data["files"][file]["masks"]

                self.data["files"][file]["nuclear_features"][f"{ch}_positive"] = []
                self.data["files"][file]["nuclear_features"][f"{ch}_frac_covered"] = []

                for cell in self.data["files"][file]["nuclear_features"]["cellID"]:
                    if cell != 0:
                        avg_binary = np.average(bin_img[mask == cell])
                        self.data["files"][file]["nuclear_features"][f"{ch}_frac_covered"].append(avg_binary)
                        if avg_binary >= frac_covered:
                            self.data["files"][file]["nuclear_features"][f"{ch}_positive"].append(True)
                        else:
                            self.data["files"][file]["nuclear_features"][f"{ch}_positive"].append(False)

    def assign_marker_class(self, n_class = 3):
        """
        Assign a class to each cell according to marker intensity using a multi-otsu function.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        for ch in tqdm(self.data["channels_info"]):
            if ch == self.data["dna_marker"]:
                continue
            img_concat = cv2.vconcat([self.data["files"][file]['working_array'][self.data["channels_info"][ch]] for file in self.data["files"]])
            thresholds = threshold_multiotsu(img_concat, n_class)
            for file in self.data["files"]:
                image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                mask = self.data["files"][file]["masks"]
                regions = np.digitize(image, bins = thresholds)
                self.data["files"][file]["nuclear_features"][f"{ch}_class"] = []
                for cell in self.data["files"][file]["nuclear_features"]["cellID"]:
                    if cell == 0:
                        continue
                    _class = stats.mode(regions[mask == cell])[0][0]
                    self.data["files"][file]["nuclear_features"][f"{ch}_class"].append(_class)

    def markerGroup(self, n_groups = 5, sample_size = None):
        """
        Assign a group to each cell according to marker intensity using KMeans.

        Parameters
        ----------
        n_groups : int, optional
            Number of groups in which to classify marker intensity. The default is 5.
        sample_size : int, optional
            Number of images to use to determine thresholds. The default is None (uses all images).
            A large number of samples (n > 10) is computationally extensive.

        Returns
        -------
        None.

        """
        if sample_size is None:
            files = [file for file in self.data["files"]]
        else:
            n_files = len(self.data["files"])
            files = [file for file in self.data["files"]]
            if n_files > sample_size:
                files = random.sample(files, sample_size)
            else:
                print(f"Given sample size ({sample_size}) is larger than (or equal to) the total number of samples ({len(files)}). Using all samples to determine thresholds.")

        for ch in tqdm(self.data["channels_info"]):
            if ch == self.data["dna_marker"]:
                continue
            img_concat = cv2.vconcat([self.data["files"][file]['working_array'][self.data["channels_info"][ch]] for file in files])
            kmeans = KMeans(n_clusters = n_groups, random_state = 0).fit(img_concat.reshape((-1, 1)))
            thresholds = kmeans.cluster_centers_.squeeze()
            for file in self.data["files"]:
                image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                mask = self.data["files"][file]["masks"]
                regions = np.digitize(image, bins = sorted(thresholds))
                self.data["files"][file]["nuclear_features"][f"{ch}_group"] = []
                for cell in self.data["files"][file]["nuclear_features"]["cellID"]:
                    if cell == 0:
                        continue
                    group = stats.mode(regions[mask == cell])[0][0]
                    self.data["files"][file]["nuclear_features"][f"{ch}_group"].append(group)

    def get_lst_features(self):
        """
        Gets list of measured features.

        Returns
        -------
        lst_fts : list
            list of features measured.

        """
        lst_fts = []

        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                lst_fts.append(ft)
            break

        return lst_fts


    def export_csv(self, filename = "raw_output.csv"):
        """
        Export data generated as CSV

        Parameters
        ----------
        filename : str, optional
            Name of output file. The default is "output.csv".

        Returns
        -------
        None.

        """
        if not filename.endswith(".csv"):
            filename = filename + ".csv"

        lst_fts = self.get_lst_features()

        dct_df = {}

        for ft in lst_fts:
            dct_df[ft] = [l for file in self.data["files"] for l in self.data["files"][file]["nuclear_features"][ft]]

        df_out = pd.DataFrame.from_dict(data = dct_df)
        df_out.to_csv(self.path_save + filename, index = False)

        print(f"CSV file saved as: {self.path_save + filename}.")


    def saveArrays(self):
        """
        Saves arrays (images and masks) in NPY format.

        Returns
        -------
        None.

        """
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        for file in tqdm(self.data["files"]):
            os.makedirs(self.path_save + "/" + file)
            np.save(self.path_save + file + "/" + file + "_wkarray.npy", self.data["files"][file]["working_array"])
            np.save(self.path_save + file + "/" + file + "_masks.npy", self.data["files"][file]["masks"])


    def saveChannelInfo(self):
        """
        Saves information of channels in JSON format.

        Returns
        -------
        None.

        """
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
        try:
            with open(self.path_save + "channels_info.json", "w") as outfile:
                json.dump(self.data["channels_info"], outfile)
            print("Channel Info saved successfully!")
        except:
            print("Ops! An error occurred while saving the info...")


    def oneClic_ngs(self, segmentation_method = "cellpose", segmentation_diameter = 30,
                    segmentation_gamma_corr = True, segmentation_gamma = 0.25,
                    segmentation_dc_scaleCorr = 1.9, dnaDots_boxSize = 10,
                    dnaDots_zoomBoxSize = 300, spEntropy_d = 5, spEntropy_zoomBoxSize = 300,
                    pos2mkr_fracCovered = 0.7, pos2mkr_thMethod = "triangle"):
        """
        An all-in-one function for running the main NGS functions.

        Parameters
        ----------
        segmentation_method : string, optional
            Nuclear segmentation method to be employed. The default is "cellpose".
        segmentation_diameter : int, optional
            Approx nuclear diamenter in px. The default is 30.
        segmentation_gamma_corr : boolean, optional
            Perform gamma correction before segmentation. The default is True.
        segmentation_gamma : float, optional
            Gamma correction value. The default is 0.25.
        segmentation_dc_scaleCorr : float, optional
            DeepCell scale correction. The default is 1.9.
        dnaDots_boxSize : int, optional
            Side of a box in pixels for minimum distance between dna dots. The default is 10.
        dnaDots_zoomBoxSize : int, optional
            Side of a zoom-in box in pixels for measuring dna dots. The default is 300.
        spEntropy_d : int, optional
            Side of a box in pixels for regional measuring of entropy. The default is 5.
        spEntropy_zoomBoxSize : int, optional
            Side of a zoom-in box in pixels for measuring spatial entropy. The default is 300.
        pos2mkr_fracCovered : float, optional
            Fraction covered of the nucleus for identifying it as positive to a marker. The default is 0.7.
        pos2mkr_thMethod : str, optional
            Marker channel thresholding method. The default is "triangle".

        Returns
        -------
        None.

        """
        # Perform nuclear segmentation
        print("Performing nuclear segmentation...")
        self.nuclear_segmentation(method = segmentation_method,
                                  diameter = segmentation_diameter,
                                  gamma_corr = segmentation_gamma_corr,
                                  gamma = segmentation_gamma,
                                  dc_scaleCorr = segmentation_dc_scaleCorr)
        print("Nuclear segmentation DONE.")

        # Measure first set of nuclear features
        print("\nMeasuring first set of nuclear features...")
        self.nuclear_features()
        print("DONE.")

        # Measure second set of nuclear features
        print("\nMeasuring second set of nuclear features...")
        self.add_nuclear_features()
        print("DONE.")

        # Find DNA dots
        print("\nFinding DNA dots...")
        self.find_dna_dots(box_size = dnaDots_boxSize, zoom_box_size = dnaDots_zoomBoxSize)
        print("DONE.")

        # Measure Spatial Entropy
        print("\nMeasuring spatial entropy...")
        self.spatial_entropy(d = spEntropy_d, zoom_box_size = spEntropy_zoomBoxSize)
        print("DONE.")

        # Identifying cells positive to markers
        print("\nIdentifying cells positive to markers...")
        self.positive2marker(frac_covered = pos2mkr_fracCovered,
                             thresh_method = pos2mkr_thMethod)
        print("DONE.")

        # Save arrays
        print("\nSaving arrays...")
        self.saveArrays()
        print("DONE.")

        # Save channel info
        print("\nSaving channel information...")
        self.saveChannelInfo()
        print("DONE.")

        print("\nSUCCESS! Segmentation and nuclear feature measurement correctly performed.")

