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
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from spatialentropy import leibovici_entropy
from sklearn.cluster import KMeans
#from deepcell.applications import NuclearSegmentation
from scipy import stats
import tifffile as tiff

#############################################
#     Functions & Classes | Segmentation    #
#############################################

def get_array_tiff(file, ext = None, stackcompression = "max"):

    metadata = {'Channels': [],
                'Axes': "TIFF",
                'ResUnit': 0,
                'XScale': 0,
                'YScale': 0}

    # import tiff and metadata
    tiffarray = tiff.imread(file)
    tiffmeta = tiff.TiffFile(file)

    # check shape
    ## if it is a z-stack, get mean or max
    if len(tiffarray.shape) == 4:
        if stackcompression == "max":
            tiffarray = np.max(tiffarray, axis=0)
        elif stackcompression == "mean":
            tiffarray = np.mean(tiffarray, axis=0)

    # force correct datatype
    tiffarray = np.array(tiffarray, dtype = "uint16")

    # get required metadata
    if ext == ".lsm":
        lsm_meta = tiffmeta.pages[0].tags[34412].value
        metadata["XScale"] = lsm_meta['VoxelSizeX']*1000000
        metadata["YScale"] = lsm_meta['VoxelSizeY']*1000000


        metadata["Channels"] = tiffmeta.pages[0].tags[34412].value['ChannelColors']['ColorNames']
    elif ext in [".tiff",".tif"]:
        xscale = tiffmeta.pages[0].tags[282].value[1]/tiffmeta.pages[0].tags[282].value[0]
        yscale = tiffmeta.pages[0].tags[283].value[1] / tiffmeta.pages[0].tags[283].value[0]
        # detect res unit
        metadata["ResUnit"] = tiffmeta.pages[0].tags[296].value.name
        if metadata["ResUnit"] == "CENTIMETER":
            xscale = xscale*(1000)
            yscale = yscale * (1000)

        metadata["XScale"] = xscale
        metadata["YScale"] = yscale
        metadata["Channels"] = list(range(tiffarray.shape[0]))
    metadata["XScale"] = round(metadata["XScale"], 3)
    metadata["YScale"] = round(metadata["YScale"], 3)

    return tiffarray, metadata


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

    # # get directory and filename etc.
    # try:
    #     metadata['Directory'] = os.path.dirname(filename)
    # except:
    #     metadata['Directory'] = 'Unknown'
    # try:
    #     metadata['Filename'] = os.path.basename(filename)
    # except:
    #     metadata['Filename'] = 'Unknown'
    # metadata['Extension'] = 'czi'
    # metadata['ImageType'] = 'czi'

    # add axes and shape information
    metadata['Axes'] = czi.axes
    # metadata['Shape'] = czi.shape
    #
    # # determine pixel type for CZI array
    # metadata['NumPy.dtype'] = str(czi.dtype)
    #
    # # check if the CZI image is an RGB image depending on the last dimension entry of axes
    # if czi.axes[-1] == 3:
    #     metadata['isRGB'] = True
    #
    metadata['Information'] = metadatadict_czi['ImageDocument']['Metadata']['Information']
    # try:
    #     metadata['PixelType'] = metadata['Information']['Image']['PixelType']
    # except KeyError as e:
    #     print('Key not found:', e)
    #     metadata['PixelType'] = None
    #
    # metadata['SizeX'] = int(metadata['Information']['Image']['SizeX'])
    # metadata['SizeY'] = int(metadata['Information']['Image']['SizeY'])
    #
    # try:
    #     metadata['SizeZ'] = int(metadata['Information']['Image']['SizeZ'])
    # except:
    #     if dim2none:
    #         metadata['SizeZ'] = None
    #     if not dim2none:
    #         metadata['SizeZ'] = 1
    #
    try:
        metadata['SizeC'] = int(metadata['Information']['Image']['SizeC'])
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
    #
    # try:
    #     metadata['SizeT'] = int(metadata['Information']['Image']['SizeT'])
    # except:
    #     if dim2none:
    #         metadata['SizeT'] = None
    #     if not dim2none:
    #         metadata['SizeT'] = 1
    #
    # try:
    #     metadata['SizeM'] = int(metadata['Information']['Image']['SizeM'])
    # except:
    #     if dim2none:
    #         metadata['SizeM'] = None
    #     if not dim2none:
    #         metadata['SizeM'] = 1
    #
    # try:
    #     metadata['SizeB'] = int(metadata['Information']['Image']['SizeB'])
    # except:
    #
    #     if dim2none:
    #         metadata['SizeB'] = None
    #     if not dim2none:
    #         metadata['SizeB'] = 1
    #
    # try:
    #     metadata['SizeS'] = int(metadata['Information']['Image']['SizeS'])
    # except:
    #     if dim2none:
    #         metadata['SizeS'] = None
    #     if not dim2none:
    #         metadata['SizeS'] = 1

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
    #
    # # try to get software version
    # try:
    #     metadata['SW-Name'] = metadata['Information']['Application']['Name']
    #     metadata['SW-Version'] = metadata['Information']['Application']['Version']
    # except KeyError as e:
    #     print('Key not found:', e)
    #     metadata['SW-Name'] = None
    #     metadata['SW-Version'] = None
    #
    # try:
    #     metadata['AcqDate'] = metadata['Information']['Image']['AcquisitionDateAndTime']
    # except KeyError as e:
    #     print('Key not found:', e)
    #     metadata['AcqDate'] = None
    #
    # try:
    #     metadata['Instrument'] = metadata['Information']['Instrument']
    # except KeyError as e:
    #     print('Key not found:', e)
    #     metadata['Instrument'] = None
    #
    # if metadata['Instrument'] is not None:
    #
    #     # get objective data
    #     try:
    #         metadata['ObjName'] = metadata['Instrument']['Objectives']['Objective']['@Name']
    #     except:
    #         metadata['ObjName'] = None
    #
    #     try:
    #         metadata['ObjImmersion'] = metadata['Instrument']['Objectives']['Objective']['Immersion']
    #     except:
    #         metadata['ObjImmersion'] = None
    #
    #     try:
    #         metadata['ObjNA'] = float(metadata['Instrument']['Objectives']['Objective']['LensNA'])
    #     except:
    #         metadata['ObjNA'] = None
    #
    #     try:
    #         metadata['ObjID'] = metadata['Instrument']['Objectives']['Objective']['@Id']
    #     except:
    #         metadata['ObjID'] = None
    #
    #     try:
    #         metadata['TubelensMag'] = float(metadata['Instrument']['TubeLenses']['TubeLens']['Magnification'])
    #     except:
    #         metadata['TubelensMag'] = None
    #
    #     try:
    #         metadata['ObjNominalMag'] = float(metadata['Instrument']['Objectives']['Objective']['NominalMagnification'])
    #     except KeyError as e:
    #         print('Key not found:', e)
    #         metadata['ObjNominalMag'] = None
    #
    #     try:
    #         metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
    #     except:
    #         metadata['ObjMag'] = None
    #
    #     # get detector information
    #     try:
    #         metadata['DetectorID'] = metadata['Instrument']['Detectors']['Detector']['@Id']
    #     except:
    #         metadata['DetectorID'] = None
    #
    #     try:
    #         metadata['DetectorModel'] = metadata['Instrument']['Detectors']['Detector']['@Name']
    #     except:
    #         metadata['DetectorModel'] = None
    #
    #     try:
    #         metadata['DetectorName'] = metadata['Instrument']['Detectors']['Detector']['Manufacturer']['Model']
    #     except:
    #         metadata['DetectorName'] = None
    #
    #     # delete some key from dict
    #     del metadata['Instrument']
    #
    # # check for well information
    #
    # metadata['Well_ArrayNames'] = []
    # metadata['Well_Indices'] = []
    # metadata['Well_PositionNames'] = []
    # metadata['Well_ColId'] = []
    # metadata['Well_RowId'] = []
    # metadata['WellCounter'] = None
    #
    # try:
    #     allscenes = metadata['Information']['Image']['Dimensions']['S']['Scenes']['Scene']
    #     for s in range(metadata['SizeS']):
    #         well = allscenes[s]
    #         metadata['Well_ArrayNames'].append(well['ArrayName'])
    #         metadata['Well_Indices'].append(well['@Index'])
    #         metadata['Well_PositionNames'].append(well['@Name'])
    #         metadata['Well_ColId'].append(well['Shape']['ColumnIndex'])
    #         metadata['Well_RowId'].append(well['Shape']['RowIndex'])
    #
    #     # count the content of the list, e.g. how many time a certain well was detected
    #     metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])
    #     # count the number of different wells
    #     metadata['NumWells'] = len(metadata['WellCounter'].keys())
    #
    # except:
    #     pass
    #     #print('Key not found: S')
    #     #print('No Scence or Well Information detected:')
    #

    # for getting binning
    #
    # try:
    #     channels = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
    #     for channel in range(len(channels)):
    #         cuch = channels[channel]
    #         metadata['Binning'].append(cuch['DetectorSettings']['Binning'])
    #
    # except KeyError as e:
    #     print('Key not found:', e)
    #     print('No Binning Found')
    #
    # del metadata['Information']
    # del metadata['Scaling']
    #
    # # close CZI file
    # czi.close()

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

    elif axes == 'TIFF':
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
    area_0 = np.array([ceil(mask_props[n]['area']) for n in range(len(mask_props))])

    core_avg_int = []
    core_total_int = []
    kernel = np.ones((3, 3), np.uint16)

    ## testing, to speeed up identifiying core mask
    to_erode = np.array([True]*len(area_0))
    to_erode[to_erode<64] = False
    to_erode_index = np.array(list(range(1, len(area_0)+1)))
    area_after = area_0.copy()
    core_mask = mask.copy()

    while any(to_erode):
        # binarize core_mask
        bin_core_mask = core_mask.copy()
        bin_core_mask[bin_core_mask > 0] = 1

        # erode binarized core mask and get the difference
        eroded_bin_mask = np.array(cv2.erode(bin_core_mask, kernel, iterations=1))
        diff_bin_core_mask = bin_core_mask - eroded_bin_mask

        # convert binarized difference to labels
        diff_core_mask = core_mask.copy()
        diff_core_mask[diff_bin_core_mask == 0] = 0

        # remove masks to be excluded
        diff_core_mask[np.isin(diff_core_mask,to_erode_index[to_erode], invert=True)] = 0

        # subtract difference from core mask
        core_mask = core_mask - diff_core_mask

        # get new core mask properties
        core_mask_props = regionprops(core_mask, intensity_image=image)
        area_before = area_after
        ids = [core_mask_props[n]['label']-1 for n in range(len(core_mask_props))]
        area_after[ids] = [ceil(core_mask_props[n]['area']) for n in range(len(core_mask_props))]

        to_erode = (np.array(area_after) != np.array(area_before))&(np.array(area_after) > np.array(area_0)/2)


    core = core_mask
    core_props = regionprops(core, intensity_image=image)
    core_avg_int = [ceil(core_props[n]['intensity_mean']) for n in range(len(core_props))]
    core_total_int = [np.sum(core_props[n]['image_intensity']) for n in range(len(core_props))]

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

def get_th_array(masks, nucleus):

    masks=masks.copy()
    nucleus=nucleus.copy()
    nucleus[masks == 0] = 0

    ignore_mask = np.zeros(masks.shape)
    ignore_mask[masks == 0] = True
    ignore_mask[masks != 0] = False
    ignore_mask = ignore_mask.astype(bool)

    bkg = Background2D(nucleus, 3, mask=ignore_mask)
    th = detect_threshold(data=nucleus, nsigma=0, mask=0, background=bkg.background)
    return th
def removenuclei(masks):

    masks = masks.copy()

    # get index of edges
    y = [3,len(masks)-3]
    x = [3,len(masks[0])-3]

    # get labels found on edges
    mask_border_x = masks[:,x]
    mask_border_y = masks[y]
    mask_border = np.append(mask_border_x.flatten(), mask_border_y.flatten())
    mask_border_seg = list(set(mask_border[mask_border>0]))

    # convert edge nuclei masks to 0
    masks[np.isin(masks, mask_border_seg)] = 0

    props = regionprops(masks)
    small_nuclei = [props[n]['label'] for n in range(len(props)) if props[n]['area'] < 25]

    masks[np.isin(masks, small_nuclei)] = 0

    return(masks)





class Segmentador(object):

    def __init__(self, indir, outdir=None, analyse_all=False, resolution=None):
        """
        Initialise Segmentador object

        Parameters
        ----------
        indir : string
            Path to experiment directory containing image file(s)
        outdir : string
            Path to output directory. If None, indir will be set as the output directory.
        analyse_all : bool
            Whether to analyse all images in directory. If false, function will request input

        Returns
        -------
        None.

        """
        # TODO: add compatibility to more formats.
        formats = [".czi", ".tiff", ".tif", ".lsm"]

        # save input dir to obj if indir exists
        indir = os.path.normpath(indir)
        if os.path.exists(indir):
            self.path_read = indir
        else:
            raise OSError(f"{indir} does not exists")

        # automatically detect supported file types
        extensions = list(set([os.path.splitext(f)[1] for f in listdir(self.path_read) if isfile(join(self.path_read, f))]))
        if any(x in formats for x in extensions):
            extensions = np.array(extensions)[[x in formats for x in extensions]]  # remove unsupported file types
            # set image_format if there is only 1 extension type
            if len(extensions) == 1:
                self.image_format = extensions[0]
            elif len(extensions) > 1:
                format_input = ""
                while format_input not in formats:
                    format_input = input(f'Multiple image formats found, please input format to analyze ({"/ ".join(extensions)}): ')
                self.image_format = format_input
        else:
            raise OSError(f"{indir} does not contain any supported image files")


        # Create data slot of dict datatype
        self.data = {}
        self.data["files"] = {}

        # get all supported files
        files = [f for f in listdir(self.path_read) if
                 isfile(join(self.path_read, f)) and f.lower().endswith(self.image_format)]

        # subset files based on user requirements
        if analyse_all == False:
            no_files = ""
            while no_files not in ["all", "one"]:
                no_files = input(f'Analyse all ({len(files)}) {self.image_format} files or select one (all/one)? ').lower()

            if no_files == "one":
                print("\n".join(files))
                new_files = ""

                while new_files not in files:
                    new_files = input("\nEnter name of file to analyse: ")

        print("Files imported:")
        for file in files:
            _file = file.replace(self.image_format, "")
            self.data["files"][_file] = {}
            self.data["files"][_file]["path"] = join(self.path_read, file)
            print(f"\t{_file}", f"(format: {self.image_format.upper()})")

            if self.image_format == ".czi":
                array, metadata, moremetadata = get_array_czi(filename=self.data["files"][_file]["path"])
                self.data["files"][_file]["array"] = array
                self.data["files"][_file]["metadata"] = metadata
                # self.data["files"][_file]["add_metadata"] = moremetadata
            # TODO: support TIFF files
            elif self.image_format in ['.tiff','.tif','.lsm']:
                array, metadata = get_array_tiff(file=self.data["files"][_file]["path"], ext = self.image_format)
                self.data["files"][_file]["array"] = array
                self.data["files"][_file]["metadata"] = metadata
        if resolution == None:
            self.check_pxScale()
        elif len(resolution)==2:
            self.check_pxScale(resolution[0], resolution[1])
        else:
            self.check_pxScale()


        # Creat out folder in the same path, and increase out_ng suffix to prevent overwrite
        outdir = indir if outdir is None else join(outdir, os.path.basename(indir))
        self.path_save = join(outdir, 'out_ng')
        if os.path.isdir(self.path_save):
            n = 1
            while os.path.isdir(self.path_save):
                self.path_save = join(outdir, f'out_ng ({n})')
                n += 1


    def set_channels(self, channels = None, marker = None):
        """
        Assign channel identity

        Parameters
        ----------
        channels : str list
            List of names of each channel in order. If None, function will request user input
        marker : string
            Name of nuclear marker

        Returns
        -------
        None.

        """


        if channels == None:
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

            self.data["dna_marker"] = ""
            while self.data["dna_marker"] not in list(self.data['channels_info'].keys()):
                self.data["dna_marker"] = input(f"\nWhich marker is the DNA marker (nuclear staining) ({'/'.join(self.data['channels_info'].keys())})? ")

        else:
            # check length of input
            firstfile = list(self.data["files"])[0]
            expected_length = len(self.data["files"][firstfile]['metadata']['Channels'])
            if len(channels) != expected_length:
                raise ValueError(f"{len(channels)} channels given when image has {expected_length}")

            # check marker input
            if marker not in channels:
                raise ValueError(f"Nuclear marker `{marker}` not found in given channels")

            self.data["channels_info"] = {}
            for n, channel in enumerate(channels):
                self.data["channels_info"][channel] = n
                self.data["dna_marker"] = marker


    def nuclear_segmentation(self, method = "cellpose", diameter = None, gamma_corr = None, dc_scaleCorr = None, GPU = False):
        """
        Perform nuclear segmentation.

        Parameters
        ----------
        method : str
            Method to segment nuclei. Currently supports "cellpose" (default)
        diameter : Integer
            Approximate nuclear diameter. The default is None.
        gamma_corr: float or None
            Gamma value to correct intensity by. By default, gamma_corr is not performed.
        dc_scaleCorr : float
            Scale correction for deepcell segmentation
        GPU : bool
            Whether to use GPU for segmentation

        Returns
        -------
        None.

        """

        self.check_pxScale()
        if method.lower() == "cellpose":
            self.seg_method = "cellpose"
            for n, file in enumerate(self.data["files"]):
                print(f"\nPerforming segmentation on file {n+1} of {len(self.data['files'])} \n")
                self.data["files"][file]['working_array'] = wk_array(self.data["files"][file]['array'],
                                                                     self.data["files"][file]['metadata']['Axes'])

                nuclei = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                if gamma_corr is not None:
                    nuclei = gammaCorrection(img = nuclei, gamma = gamma_corr)
                self.data["files"][file]["masks"], self.data["files"][file]["flows"] = _cellpose(nuclei,
                                                                                     diameter = diameter,
                                                                                       GPU = GPU)
                self.data['files'][file]["masks"] = removenuclei(self.data['files'][file]["masks"])
                nucleus = self.data["files"][file]['working_array'][
                    self.data["channels_info"][self.data["dna_marker"]]].copy()
                self.data["files"][file]["th_array"] = get_th_array(self.data["files"][file]["masks"], nucleus)

        elif method.lower() == "deepcell":
            self.seg_method = "deepcell"
            for file in tqdm(self.data["files"]):
                self.data["files"][file]['working_array'] = wk_array(self.data["files"][file]['array'],
                                                                     self.data["files"][file]['metadata']['Axes'])

                nuclei = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
                if gamma_corr is not None:
                    nuclei = gammaCorrection(img = nuclei, gamma = gamma_corr)
                if dc_scaleCorr == None:
                    dc_scaleCorr = 1
                self.data["files"][file]["masks"] = _deepcell(image = nuclei,
                                                              scale = self.data["files"][file]['metadata']['XScale'] * dc_scaleCorr)
                self.data['files'][file]["masks"] = removenuclei(self.data['files'][file]["masks"])
                nucleus = self.data["files"][file]['working_array'][
                    self.data["channels_info"][self.data["dna_marker"]]].copy()
                self.data["files"][file]["th_array"] = get_th_array(self.data["files"][file]["masks"], nucleus)

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
                            xscale = float(input("Enter resolution for X axis (micrometer/pixel): "))
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
                            yscale = float(input("Enter resolution for Y axis (micrometer/pixel): "))
                            self.data["files"][file]['metadata']['YScale'] = yscale
                            yflag = False
                            break
                        except:
                            print("Invalid input! Try again...\n")
                self.data["files"][file]['metadata']['YScale'] = yscale


    def nuclear_features(self):
        """
        Measure first pool of nuclear features.

        Returns
        -------
        None.

        """

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

                    self.data["files"][file]["nuclear_features"]["angle"].append(round(p["orientation"],3))

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
            outlist = dict((key,0) for key in self.data["files"][file]["nuclear_features"]["cellID"])

            masks = self.data["files"][file]["masks"].copy()

            nucleus = self.data["files"][file]['working_array'][
                self.data["channels_info"][self.data["dna_marker"]]].copy()
            nucleus[masks == 0] = 0

            th = self.data["files"][file]["th_array"]


            ignore_mask = np.zeros(masks.shape)
            ignore_mask[masks == 0] = True
            ignore_mask[masks != 0] = False
            ignore_mask = ignore_mask.astype(bool)

            peak_tb = find_peaks(data=nucleus, threshold=th, mask=ignore_mask, box_size=box_size)

            peak_df = peak_tb.to_pandas()
            peak_df['x_round'] = 10 * np.ceil(peak_df['x_peak'].to_numpy()/10)
            peak_df['y_round'] = 10 * np.ceil(peak_df['y_peak'].to_numpy() / 10)

            round_peak_df = peak_df[["x_round","y_round"]]
            round_peak_df = round_peak_df.drop_duplicates()

            merged_peak_df = peak_df.iloc[round_peak_df.index,]
            mapped_peaks = masks[merged_peak_df["y_peak"].to_list(), merged_peak_df["x_peak"].to_list()]
            unique, counts = np.unique(mapped_peaks, return_counts=True)
            outlist.update(zip(unique, counts))


            self.data["files"][file]["nuclear_features"]["dna_peaks"] = list(outlist.values())




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
            outlist = dict((key, 0) for key in self.data["files"][file]["nuclear_features"]["cellID"])
            area_outlist = dict((key, 0) for key in self.data["files"][file]["nuclear_features"]["cellID"])

            masks = self.data["files"][file]["masks"].copy()

            nucleus = self.data["files"][file]['working_array'][
                self.data["channels_info"][self.data["dna_marker"]]].copy()
            nucleus[masks == 0] = 0

            th = self.data["files"][file]["th_array"]

            ignore_mask = np.zeros(masks.shape)
            ignore_mask[masks == 0] = True
            ignore_mask[masks != 0] = False
            ignore_mask = ignore_mask.astype(bool)

            sigma = 3.0 * gaussian_fwhm_to_sigma

            kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
            kernel.normalize()
            segm = detect_sources(data=nucleus, threshold=th*1.15, npixels=7, kernel=kernel, mask=ignore_mask)

            cat = SourceCatalog(nucleus, segm)
            columns = ['label', 'xcentroid', 'ycentroid', 'area']
            dots_df = cat.to_table(columns).to_pandas()
            dots_df['xcentroid'] = dots_df[['xcentroid']].astype(int)
            dots_df['ycentroid'] = dots_df['ycentroid'].astype(int)
            dots_df["cell"] = masks[dots_df["ycentroid"].to_list(), dots_df["xcentroid"].to_list()]
            dots_df = dots_df[dots_df["cell"] != 0]  # ensure removal of dots from non-masks

            unique, counts = np.unique(dots_df["cell"], return_counts=True)
            outlist.update(zip(unique, counts))
            self.data["files"][file]["nuclear_features"]["dna_dots"] = list(outlist.values())

            median_area = dots_df.groupby('cell')['area'].median()
            median_area = median_area * (self.data["files"][file]['metadata']['XScale'] * self.data["files"][file]['metadata']['YScale'])
            area_outlist.update(zip(median_area.index.to_numpy().astype(int), median_area.values))

            self.data["files"][file]["nuclear_features"]["dna_dots_size_median"] = np.round(list(area_outlist.values()),3)


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
            outlist = dict((key, 0) for key in self.data["files"][file]["nuclear_features"]["cellID"])

            masks = self.data["files"][file]["masks"].copy()
            nucleus = self.data["files"][file]['working_array'][
                self.data["channels_info"][self.data["dna_marker"]]].copy()
            nucleus[masks == 0] = 0



            signal_coord = np.argwhere(masks > 0)
            signal_pd = pd.DataFrame(signal_coord, columns=['x_coord','y_coord'])
            signal_pd['cellID'] = masks[signal_pd['x_coord'],signal_pd['y_coord']]
            signal_pd['intensity'] = nucleus[signal_pd['x_coord'], signal_pd['y_coord']]
            signal_pd["rank"] = signal_pd.groupby("cellID")["intensity"].rank("min")
            signal_max_grouped = signal_pd.groupby("cellID", as_index=False)["rank"].max()
            signal_max_grouped = signal_max_grouped.rename(columns={"rank": "maxrank"})
            signal_pd = pd.merge(signal_pd,signal_max_grouped,on='cellID',how='left')
            signal_pd['group'] = (10*signal_pd['rank']/signal_pd['maxrank']).apply(np.floor)
            signal_pd['group'] = signal_pd['group'] + 1
            out = signal_pd.groupby('cellID').apply(lambda x: leibovici_entropy(np.array(x[["y_coord","x_coord"]]), np.intc(x["group"]), d=5).entropy)

            outlist.update(zip(out.index.to_numpy().astype(int), out.values))

            self.data["files"][file]["nuclear_features"]["spatial_entropy"] = np.round(list(outlist.values()),3)


    def markerGroup(self, n_groups = 5):
        """
        Assign a group to each cell according to marker intensity using KMeans.

        Parameters
        ----------
        n_groups : int, optional
            Number of groups in which to classify marker intensity. The default is 5.

        Returns
        -------
        None.

        """
        files = [file for file in self.data["files"]]
        for ch in tqdm(self.data["channels_info"]):
            if ch == self.data["dna_marker"]:
                continue
            img_concat = cv2.vconcat([self.data["files"][file]['working_array'][self.data["channels_info"][ch]] for file in files])
            img_flatten = img_concat.flatten()
            img_sampled = np.random.choice(img_flatten, replace=False, size = self.data["files"][files[0]]['masks'].size)
            kmeans = KMeans(n_clusters = n_groups, random_state = 0).fit(img_sampled.reshape((-1, 1)))
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
            lut = np.zeros_like(idx)
            lut[idx] = np.arange(n_groups)
            for file in self.data["files"]:
                ch_int = np.array(self.data["files"][file]["nuclear_features"][f"avg_intensity_{ch}"])
                self.data["files"][file]["nuclear_features"][f"{ch}_group"] = list(lut[list(kmeans.predict(ch_int.reshape((-1,1))))])


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


    def export_csv(self, filename = "output.csv"):
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
        df_out.to_csv(join(self.path_save, filename), index = False)

        print(f"CSV file saved as: {join(self.path_save, filename)}.")


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
            os.makedirs(join(self.path_save , file))
            np.save(join(self.path_save, file, file + "_wkarray.npy"), self.data["files"][file]["working_array"])
            np.save(join(self.path_save, file, file + "_masks.npy"), self.data["files"][file]["masks"])


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
            with open(join(self.path_save, "channels_info.json"), "w") as outfile:
                json.dump(self.data["channels_info"], outfile)
            print("Channel Info saved successfully!")
        except:
            print("Ops! An error occurred while saving the info...")

