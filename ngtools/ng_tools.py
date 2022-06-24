# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:52:32 2021

@author: gabrielemilioherreraoropeza
"""

import sys, os, subprocess, json, random, statistics, operator, itertools
import warnings

warnings.filterwarnings("ignore")


from os import listdir
from os.path import isfile, join, isdir
from collections import Counter
from math import pi, sqrt, log10, ceil



### --- Define "install" to install missing modules

def install(package):
    """
    Installs any module that is not currently installed in the system
    """
    print(f"Installing {package}...", end = " ")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("DONE")


### --- Import modules

try:
    import czifile as zis
except ImportError:
    install('czifile')
    import czifile as zis

try:
    import numpy as np
except ImportError:
    install('numpy')
    import numpy as np

try:
    import xmltodict
except ImportError:
    install('xmltodict')
    import xmltodict

try:
    from cellpose import models, plot
except ImportError:
    install('cellpose')
    from cellpose import models, plot

try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    from skimage.measure import regionprops
    from skimage.filters import (threshold_otsu, threshold_isodata, threshold_li, threshold_mean,
                                 threshold_minimum, threshold_triangle, threshold_yen, threshold_sauvola,
                                 gaussian, threshold_multiotsu)
    from skimage import exposure, img_as_float
except ImportError:
    install("scikit-image")
    from skimage.measure import regionprops
    from skimage.filters import (threshold_otsu, threshold_isodata, threshold_li, threshold_mean,
                                 threshold_minimum, threshold_triangle, threshold_yen, threshold_sauvola,
                                 gaussian, threshold_multiotsu)
    from skimage import exposure, img_as_float

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    import plotly.express as px
except ImportError:
    install('plotly')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    import plotly.express as px

try:
    from tqdm import tqdm
except ImportError:
    install('tqdm')
    from tqdm import tqdm

try:
    import cv2
except ImportError:
    install('opencv-python')
    import cv2

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    from photutils.detection import find_peaks
    from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
    from photutils.background import Background2D
except ImportError:
    install("photutils")
    from photutils.detection import find_peaks
    from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
    from photutils.background import Background2D

try:
    from astropy.convolution import Gaussian2DKernel
    from astropy.stats import gaussian_fwhm_to_sigma
except ImportError:
    from astropy.convolution import Gaussian2DKernel
    from astropy.stats import gaussian_fwhm_to_sigma

try:
    from spatialentropy import leibovici_entropy
except ImportError:
    install("spatialentropy")
    from spatialentropy import leibovici_entropy

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    install("scikit-learn")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

try:
    import umap.umap_ as umap
except ImportError:
    install("umap-learn")
    import umap.umap_ as umap

try:
    from deepcell.applications import NuclearSegmentation
except ImportError:
    pass
#    install("deepcell")
#    from deepcell.applications import NuclearSegmentation

try:
    from PIL import Image, ImageEnhance
except ImportError:
    install("Pillow")
    from PIL import Image, ImageEnhance

try:
    from scipy import stats
except ImportError:
    install('scipy')
    from scipy import stats

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns


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

    app = NuclearSegmentation()
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



#########################################
#     Functions & Classes | Analysis    #
#########################################

def linearReg_correction(sel_df, df, x, y):
    """
    Perform linear regression and correct selection

    Parameters
    ----------
    sel_df : pandas DataFrame
        DataFrame generated from selection.
    df : pandas DataFrame
        Original DataFrame.
    x : string
        Feature selected for X axis.
    y : string
        Feature Selected for Y axis.

    Returns
    -------
    temp_df : pandas DataFrame
        Linear-regression corrected DataFrame.

    """

    disp = float(input('\nEnter desired dispersion percentage (e.g.: 3): '))

    X = [log10(i) if i != 0 else 0 for i in sel_df[x]]
    Y = [log10(j) if j != 0 else 0 for j in sel_df[y]]

    m, b, r, p, s = stats.linregress(X, Y)

    poly1d_fn = np.poly1d(np.array([m,b]))

    flag = True
    for index, row in df.iterrows():
        if row[x] != 0:
            xx = log10(row[x])
        else:
            xx = 0
        if row[y] != 0:
            yy = log10(row[y])
        else:
            yy = 0
        prediction = (m * xx) + b
        pe = (abs((yy - prediction))/yy) * 100
        if pe <= disp:
            if flag == True:
                temp_df = df[(df['cellID'] == row['cellID']) & (df['imageID'] == row['imageID'])].copy()
                flag = False
            elif flag == False:
                temp_df = temp_df.append(df[(df['cellID'] == row['cellID']) & (df['imageID'] == row['imageID'])])

    print('\nf(x) = {0}x + {1}'.format(round(m, 3), round(b, 3)))
    print('R² = ' + str(round(r**2, 3)))
    print('std err = ' + str(round(s, 3)))
    plt.plot(X, Y, 'yo', X, poly1d_fn(X), '--k')
    plt.title('Selected Cells')
    plt.show()

    X2 = [log10(i) for i in temp_df[x]]
    Y2 = [log10(j) for j in temp_df[y]]

    m, b, r, p, s = stats.linregress(X2, Y2)
    poly1d_fn = np.poly1d(np.array([m,b]))

    print('f(x) = {0}x + {1}'.format(round(m, 3), round(b, 3)))
    print('R² = ' + str(round(r**2, 3)))
    print('std err = ' + str(round(s, 3)))
    plt.plot(X2, Y2, 'yo', X2, poly1d_fn(X2), '--k')
    plt.title('Selected Cells CORRECTED')
    plt.show()

    return temp_df


def get_lst_features(df):
    """
    Gets list of measured features.

    Returns
    -------
    lst_fts : list
        list of features measured.

    """
    lst_fts = []

    for ft in df.columns:
        if not ft in ["cellID", "imageID"]:
            lst_fts.append(ft)

    return lst_fts


def get_threshold(image, thresh_option):
    """
    Obtain threshold from image

    Parameters
    ----------
    image : array-like image
        Intensity image.
    thresh_option : str
        Thresholding option.

    Returns
    -------
    th : float
        Threshold value.

    """
    ### --- Otsu
    if thresh_option.lower() == 'otsu':
        th = threshold_otsu(image)

    ### --- Isodata
    elif thresh_option.lower() == 'isodata':
        th = threshold_isodata(image)

    ### --- Li
    elif thresh_option.lower() == 'li':
        th = threshold_li(image)

    ### --- Mean
    elif thresh_option.lower() == 'mean':
        th = threshold_mean(image)

    ### --- Minimum
    elif thresh_option.lower() == 'minimum':
        th = threshold_minimum(image)

    ### --- Triangle
    elif thresh_option.lower() == 'triangle':
        th = threshold_triangle(image)

    ### --- Yen
    elif thresh_option.lower() == 'yen':
        th = threshold_yen(image)

    ### --- Sauvola
    elif thresh_option.lower() == 'sauvola':
        th = threshold_sauvola(image)

    return th


def custom_th_img(image, th):
    """
    Generates threshold image.

    Parameters
    ----------
    image : array-like image
        Intensity image.
    th : float
        Threshold value.

    Returns
    -------
    thresh_img : array-like image
        Boolean image.

    """
    thresh_img = image > th

    return thresh_img


def roundValues(lst):
    """
    Generate list with rounded values.

    Parameters
    ----------
    lst : list
        list contaning values.

    Raises
    ------
    ValueError
        if any value in list is negative

    Returns
    -------
    round_list : list
        list containing rounded values.

    """

    if all(i <= 1 for i in lst) and all(i >= 0 for i in lst):

        temp_lst = []
        for l in lst:
            temp_lst.append(len(str(l)))

        size = statistics.mode(temp_lst)
        scale = round(size * 2 / 3)

        round_list = []
        for l in lst:
            round_list.append(round(l, scale))

        return round_list

    elif any(i < 0 for i in lst):

        raise ValueError("For normalisation of intensities there should not be negative values!")

    else:

        temp_lst = []
        for l in lst:
            if l >= 1:
                temp_lst.append(len(str(l)))
            else:
                temp_lst.append(-len(str(l+1)))

        size = statistics.mode(temp_lst)
        size = abs(size)
        scale = round(size * 2 / 3) - 1

        round_list = []
        for l in lst:
            round_list.append(round(l, -scale))

        return round_list


def dct4positive2marker(df):
    """
    Generates dictionary for population_positive2marker function.

    Parameters
    ----------
    df : pandas DataFrame
        NG generated DataFrame.

    Returns
    -------
    positive2marker_dict : dictionary
        Dict containing imageID and cell numbers as keys.

    """
    positive2marker_dict = {}
    for index, row in df.iterrows():
        if not row["imageID"] in positive2marker_dict:
            positive2marker_dict[row["imageID"]] = {}
        if not row["cellID"] in positive2marker_dict[row["imageID"]]:
            positive2marker_dict[row["imageID"]][row["cellID"]] = None

    return positive2marker_dict


def check_features(df, lst):
    df_cols = list(df.columns)
    flag = True
    for l in lst:
        if not l in df_cols:
            flag = False
            break
    return flag, l


def plot_umap(df, features, feature4cmap = None, scale = None, show_markers = False, lst_markers = None, random_state = False, size = 10):

    df_notna = df.copy()

    for key in features:
        df_notna = df_notna[df_notna[key].notna()]

    x = df_notna.reindex(columns = features).values

    if not random_state:
        reducer = umap.UMAP()
    else:
        try:
            random_state = int(random_state)
        except:
            raise ValueError("Ops! 'random_state' should be an integer.")

        reducer = umap.UMAP(random_state = random_state)

    scaled_data = StandardScaler().fit_transform(x)
    embedding = reducer.fit_transform(scaled_data)

    principalDf = pd.DataFrame(data = embedding, columns = ['UMAP 1', 'UMAP 2'])

    finalDf = pd.concat([principalDf, df_notna], axis = 1)

    if feature4cmap != None and not show_markers:

        feature_vals = finalDf[feature4cmap].to_list()

        f = go.FigureWidget([go.Scatter(y = finalDf['UMAP 1'],
                                        x = finalDf['UMAP 2'],
                                        mode = 'markers',
                                        marker = dict(color = feature_vals,
                                                      colorscale = "viridis_r",
                                                      cmax = scale[1] if scale != None else max(feature_vals),
                                                      cmin = scale[0] if scale != None else min(feature_vals),
                                                      colorbar = dict(title = "ColorBar"),
                                                      size = size
                                                      )
                                        )
                             ]
                            )
    elif feature4cmap == None and show_markers:

        markers_vals = []
        for index, row in finalDf.iterrows():
            str_v = ""
            for mkr in lst_markers:
                try:
                    val = row[f"{mkr}_positive_avgMethod"]
                except:
                    try:
                        val = row[f"{mkr}_positive"]
                    except:
                        raise ValueError("Please run function for identifying cells positive to markers.")
                if val:
                    if str_v.endswith("+"):
                        str_v += f"_{mkr}+"
                    else:
                        str_v += f"{mkr}+"
            markers_vals.append(str_v)

        finalDf["markers_vals"] = [mkr if len(mkr) > 0 else "None" for mkr in markers_vals]

        f = go.FigureWidget()
        finalDf_index = {}

        for g in np.unique(finalDf["markers_vals"]):

            temp_subset = finalDf[finalDf["markers_vals"] == g]

            # Add index ID to finalDf_index dictionary
            n = 0
            for index, row in temp_subset.iterrows():
                finalDf_index[f"{row['markers_vals']}_{n}"] = index
                n += 1

            # Generate scatter
            f.add_scatter(y = temp_subset['UMAP 1'],
                          x = temp_subset['UMAP 2'],
                          mode = "markers",
                          name = g
                          )

    else:
        raise ValueError("Both 'feature4cmap' and 'show_markers' were given. Just give one.")

    f.update_layout(
        title = "UMAP",
        xaxis = dict(title = "UMAP 2"),
        yaxis = dict(title = "UMAP 1"),
        autosize = True,
        hovermode = 'closest',
        template = "plotly_white"
    )

    t = go.FigureWidget([go.Table(
        header = dict(values = df.columns,
                      fill = dict(color='#C2D4FF'),
                      align = ['left'] * 5),
        cells = dict(values=[df[col] for col in df.columns],
                     fill = dict(color='#F5F8FF'),
                     align = ['left'] * 5))])

    scatter = f.data

    table_points = {}

    def flatten(t):
        return [item for sublist in t for item in sublist]

    def selection_fn(trace,points,selector):
        if show_markers:
            table_points[trace.name] = [finalDf_index[f"{trace.name}_{val}"] for val in points.point_inds]
        else:
            table_points[trace.name] = points.point_inds
        t.data[0].cells.values = [finalDf.reindex(index = flatten(table_points.values()))[col] for col in finalDf.columns if not any(col in s for s in ['UMAP 1', 'UMAP 2', 'markers_vals'])]

    for n in range(len(scatter)):
        scatter[n].on_selection(selection_fn)

    return f, t


def plot_tsne(df, features, feature4cmap = None, show_markers = False, lst_markers = None, size = 10, scale = None):

    df_notna = df.copy()

    for key in features:
        df_notna = df_notna[df_notna[key].notna()]

    x = df_notna.reindex(columns = features).values

    tsne = TSNE(n_components = 2,
                #verbose = 1,
                perplexity = 30,
                n_iter = 1000)
    tsne_results = tsne.fit_transform(x)

    principalDf = pd.DataFrame(data = tsne_results, columns = ['tSNE 1', 'tSNE 2'])

    finalDf = pd.concat([principalDf, df_notna], axis = 1)

    if feature4cmap != None and not show_markers:

        feature_vals = finalDf[feature4cmap].to_list()

        f = go.FigureWidget([go.Scatter(y = finalDf['tSNE 1'],
                                        x = finalDf['tSNE 2'],
                                        mode = 'markers',
                                        marker = dict(color = feature_vals,
                                                      colorscale = "viridis_r",
                                                      cmax = scale[1] if scale != None else min(feature_vals),
                                                      cmin = scale[0] if scale != None else min(feature_vals),
                                                      colorbar = dict(title = "ColorBar"),
                                                      size = size
                                                      )
                                        )
                             ]
                            )
    elif feature4cmap == None and show_markers:

        markers_vals = []
        for index, row in finalDf.iterrows():
            str_v = ""
            for mkr in lst_markers:
                try:
                    val = row[f"{mkr}_positive_avgMethod"]
                except:
                    try:
                        val = row[f"{mkr}_positive"]
                    except:
                        raise ValueError("Please run function for identifying cells positive to markers.")
                if val:
                    if str_v.endswith("+"):
                        str_v += f"_{mkr}+"
                    else:
                        str_v += f"{mkr}+"
            markers_vals.append(str_v)


        finalDf["markers_vals"] = [mkr if len(mkr) > 0 else "None" for mkr in markers_vals]

        f = go.FigureWidget()
        finalDf_index = {}

        for g in np.unique(finalDf["markers_vals"]):

            temp_subset = finalDf[finalDf["markers_vals"] == g]

            # Add index ID to finalDf_index dictionary
            n = 0
            for index, row in temp_subset.iterrows():
                finalDf_index[f"{row['markers_vals']}_{n}"] = index
                n += 1

            # Generate scatter
            f.add_scatter(y = temp_subset['tSNE 1'],
                          x = temp_subset['tSNE 2'],
                          mode = "markers",
                          name = g
                          )


    else:
        raise ValueError("Both 'feature4cmap' and 'show_markers' were given. Just give one.")

    f.update_layout(
        title = "tSNE",
        xaxis = dict(title = "tSNE 2"),
        yaxis = dict(title = "tSNE 1"),
        autosize = True,
        hovermode = 'closest',
        template = "plotly_white"
    )

    t = go.FigureWidget([go.Table(
        header = dict(values = df.columns,
                      fill = dict(color='#C2D4FF'),
                      align = ['left'] * 5),
        cells = dict(values=[df[col] for col in df.columns],
                     fill = dict(color='#F5F8FF'),
                     align = ['left'] * 5))])

    scatter = f.data

    table_points = {}

    def flatten(t):
        return [item for sublist in t for item in sublist]

    def selection_fn(trace,points,selector):
        if show_markers:
            table_points[trace.name] = [finalDf_index[f"{trace.name}_{val}"] for val in points.point_inds]
        else:
            table_points[trace.name] = points.point_inds
        t.data[0].cells.values = [finalDf.reindex(index = flatten(table_points.values()))[col] for col in finalDf.columns if not any(col in s for s in ['tSNE 1', 'tSNE 2', 'markers_vals'])]

    for n in range(len(scatter)):
        scatter[n].on_selection(selection_fn)

    return f, t


def plot_pca(df, features, feature4cmap = None, show_markers = False, lst_markers = None, size = 10, scale = None):

    df_notna = df.copy()

    for key in features:
        df_notna = df_notna[df_notna[key].notna()]

    x = df_notna.reindex(columns = features).values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])

    finalDf = pd.concat([principalDf, df_notna], axis = 1)

    if feature4cmap != None and not show_markers:

        feature_vals = finalDf[feature4cmap].to_list()

        f = go.FigureWidget([go.Scatter(y = finalDf['Principal Component 1'],
                                        x = finalDf['Principal Component 2'],
                                        mode = 'markers',
                                        marker = dict(color = feature_vals,
                                                      colorscale = "plasma",
                                                      cmax = scale[1] if scale != None else min(feature_vals),
                                                      cmin = scale[0] if scale != None else min(feature_vals),
                                                      size  = size,
                                                      colorbar = dict(title = "ColorBar")
                                                      )
                                        )
                             ]
                            )
    elif feature4cmap == None and show_markers:

        markers_vals = []
        for index, row in finalDf.iterrows():
            str_v = ""
            for mkr in lst_markers:
                try:
                    val = row[f"{mkr}_positive_avgMethod"]
                except:
                    try:
                        val = row[f"{mkr}_positive"]
                    except:
                        raise ValueError("Please run function for identifying cells positive to markers.")
                if val:
                    if str_v.endswith("+"):
                        str_v += f"_{mkr}+"
                    else:
                        str_v += f"{mkr}+"
            markers_vals.append(str_v)


        finalDf["markers_vals"] = [mkr if len(mkr) > 0 else "None" for mkr in markers_vals]

        f = go.FigureWidget()
        finalDf_index = {}

        for g in np.unique(finalDf["markers_vals"]):

            temp_subset = finalDf[finalDf["markers_vals"] == g]


            # Add index ID to finalDf_index dictionary
            n = 0
            for index, row in temp_subset.iterrows():
                finalDf_index[f"{row['markers_vals']}_{n}"] = index
                n += 1

            # Generate scatter
            f.add_scatter(y = temp_subset['Principal Component 1'],
                          x = temp_subset['Principal Component 2'],
                          mode = "markers",
                          name = g
                          )


    else:
        raise ValueError("Both 'feature4cmap' and 'show_markers' were given. Just give one.")

    f.update_layout(
        title = "Principal Component Analysis",
        xaxis = dict(title = "Principal Component 2"),
        yaxis = dict(title = "Principal Component 1"),
        autosize = True,
        hovermode = 'closest',
        template = "plotly_white"
    )

    t = go.FigureWidget([go.Table(
        header = dict(values = df.columns,
                      fill = dict(color='#C2D4FF'),
                      align = ['left'] * 5),
        cells = dict(values=[df[col] for col in df.columns],
                     fill = dict(color='#F5F8FF'),
                     align = ['left'] * 5))])

    scatter = f.data

    table_points = {}

    def flatten(t):
        return [item for sublist in t for item in sublist]

    def selection_fn(trace,points,selector):
        if show_markers:
            table_points[trace.name] = [finalDf_index[f"{trace.name}_{val}"] for val in points.point_inds]
        else:
            table_points[trace.name] = points.point_inds
        t.data[0].cells.values = [finalDf.reindex(index = flatten(table_points.values()))[col] for col in finalDf.columns if not any(col in s for s in ['Principal Component 1', 'Principal Component 2', 'markers_vals'])]

    for n in range(len(scatter)):
        scatter[n].on_selection(selection_fn)

    return f, t


def lin_int(x1, y1, x2, y2, x):
    """
    Linear Interpolation function.

    Parameters
    ----------
    x1 : float
        lower x value.
    y1 : float
        lower y value.
    x2 : float
        higher x value.
    y2 : float
        higher y value.
    x : float
        desired x value.

    Returns
    -------
    y : float
        desired y value.

    """
    y = y1 + ((x - x1) * ((y2 - y1)/(x2 - x1)))

    return y


def removeOutliers_keepSuspected(df):
    """
    Removes outliers but keeps suspected outliers.

    Parameters
    ----------
    df : pandas DataFrame
        NG generated DataFrame.

    Returns
    -------
    lower_th : float
        lower threshold.
    higher_th : float
        higher threshold.

    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1
    lower_th = Q1 - 3 * IQR
    higher_th = Q3 + 3 * IQR

    return lower_th, higher_th


def indexConversion(df, uniqID_dct):

    new_dct = {}
    for key in uniqID_dct.keys():
        imageID = uniqID_dct[key]["imageID"]
        cellID = uniqID_dct[key]["cellID"]
        index_ = df.index[(df["imageID"] == imageID) & (df["cellID"] == cellID)].to_list()[0]
        new_dct[key] = index_

    return new_dct


def umap_value(data, features, feature, value, size = 10, color = "red", figscale = 2, random_state = 42):

    # TODO: Move to inside of analysis class and and add self.UMAP to not need to re-run UMAP

    """
    UMAP plot that highlights specific value of a feature.

    Parameters
    ----------
    data : pandas DataFrame
        NG-generated DataFrame.
    features : list
        List of features to consider for dimension reduction.
    feature : string
        Feature to consider for value to highlight in plot.
    value : string or float. Depends on feature.
        Value of input feature to highlight in plot.
    size : int, optional
        Dot size. The default is 10.
    color : string, optional
        Dot color. The default is "red".
    figscale : float, optional
        Scale of figure. The default is 2.
    random_state : int, optional
        Random state for UMAP reproducibility. The default is 42.

    Raises
    ------
    ValueError
        Raises error if random_state is not an integer.

    Returns
    -------
    fig : axes
        UMAP plot highlighting input value.

    """
    df_notna = data.copy()

    for key in features:
        df_notna = df_notna[df_notna[key].notna()]

    x = df_notna.reindex(columns = features).values

    if not random_state:
        reducer = umap.UMAP()
    else:
        try:
            random_state = int(random_state)
        except:
            raise ValueError("Ops! 'random_state' should be an integer.")

        reducer = umap.UMAP(random_state = random_state)

    scaled_data = StandardScaler().fit_transform(x)
    embedding = reducer.fit_transform(scaled_data)

    principalDf = pd.DataFrame(data = embedding, columns = ['UMAP 1', 'UMAP 2'])

    finalDf = pd.concat([principalDf, df_notna], axis = 1)

    fig, ax = plt.subplots(figsize = (6.4 * figscale, 4.8 * figscale))

    ax.scatter(finalDf["UMAP 2"],
               finalDf["UMAP 1"],
               c = [color if v == value else "grey" for v in finalDf[feature]],
               s = size,
               alpha = 0.5
               )

    ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title(f"UMAP | {feature} : {value}")
    plt.tight_layout()

    return fig



class NuclearGame_Analysis(object):

    def __init__(self, arg, multi = False, dna_marker = "DAPI"):
        """
        Start NuclearGame Analysis

        Parameters
        ----------
        arg : str
            Path to CSV file generated in segmentation.

        Raises
        ------
        ValueError
            If path to CSV file is not correct.

        Returns
        -------
        None.

        """
        valid_formats = (".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt", ".csv")
        if not arg.lower().endswith(valid_formats):
            raise ValueError("Ops! Invalid format...")

        if multi == False:
            self.path2csv = arg
            self.path_dir = os.path.dirname(self.path2csv) + "/"

            # Read data as DataFrame
            print("Reading CSV file...", end = "  ")
            self.df_raw = pd.read_csv(self.path2csv)
            print("DONE")

            # Create metadata dictionary
            self.metadata = {}
            self.metadata["files"] = {}

            # Add image and masks arrays to metadata dictionary
            print("Reading array files...", end = "  ")
            for file in set(self.df_raw["imageID"]):
                self.metadata["files"][file] = {}
                self.metadata["files"][file]["working_array"] = np.load(self.path_dir + file + "/" + file + "_wkarray.npy")
                self.metadata["files"][file]["masks"] = np.load(self.path_dir + file + "/" + file + "_masks.npy")
            print("DONE")

            # Read channel info
            print("Reading channel info...", end = "  ")
            with open(self.path_dir + "channels_info.json") as json_file:
                self.metadata["channel_info"] = json.load(json_file)
            print("DONE")

            # Ask for DNA marker
            while True:
                dna_marker = input(f"\nWhich marker is the DNA marker (nuclear staining) ({'/'.join(self.metadata['channel_info'].keys())})? ")
                if dna_marker in list(self.metadata['channel_info'].keys()):
                    self.dna_marker = dna_marker
                    break
                else:
                    print(f"Ops! '{dna_marker}' is not a marker, try again...")

            # Print number of initial cells
            print(f"\nStarting analysis with {len(self.df_raw)} cells.")
        elif multi == True:
            self.dna_marker = dna_marker
            excel_suffixes = (".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt")
            if arg.endswith(".csv"):
                multi_file = pd.read_csv(arg, header = None, names = ["path", "filename"])
            elif arg.endswith(excel_suffixes):
                multi_file = pd.read_excel(arg, header = None, names = ["path", "filename"], engine='openpyxl')
            for index, row in multi_file.iterrows():
                csv = row["path"]
                indir = os.path.dirname(row["path"]) + "/"
                csv_name = row["path"].split("/")[-1]
                print(f"\nLoading {csv_name} data:")

                # Read data as DataFrame
                print("\tReading CSV file...", end = " ")
                df = pd.read_csv(csv)
                print("DONE")
                # Add image and masks arrays to metadata dictionary
                print("\tReading array files...", end = " ")
                if index == 0:
                        self.metadata = {}
                        self.metadata["files"] = {}
                for file in set(df["imageID"]):
                    fileid = str(index) + file
                    self.metadata["files"][fileid] = {}
                    self.metadata["files"][fileid]["working_array"] = np.load(indir + file + "/" + file + "_wkarray.npy")
                    self.metadata["files"][fileid]["masks"] = np.load(indir + file + "/" + file + "_masks.npy")
                print("DONE")

                df['imageID'] = str(index) + df['imageID']
                df['file'] = row["filename"]
                if index == 0:
                    self.df_raw = df
                else:
                    self.df_raw = self.df_raw.append(df)


                # Read channel info
                print("\tReading channel info...", end = " ")
                with open(indir + "channels_info.json") as json_file:
                    self.metadata["channel_info"] = json.load(json_file)
                print("DONE")

                self.df_raw =  self.df_raw.reset_index(drop=True)


    def population_positive2marker(self, thresh_method = "triangle", frac_covered = 0.8, df = None, show_plot = False):
        """
        Identifies cells positive to markers using the whole population as reference.

        Parameters
        ----------
        thresh_method : str, optional
            Thresholding method. The default is "triangle".
        frac_covered : float, optional
            Fraction of the nuclei covered by marker mask to identify the cell as positive to the marker. The default is 0.8.
        df : pandas DataFrame, optional
            NG generated DataFrame. The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        for ch in self.metadata["channel_info"].keys():
            merged_files = []
            merged_masks = []

            if ch == self.dna_marker:
                continue

            for file in self.metadata["files"]:
                merged_files.append(self.metadata["files"][file]["working_array"][self.metadata["channel_info"][ch]])
                merged_masks.append(self.metadata["files"][file]["masks"])
            img_merge = cv2.vconcat(merged_files)
            masks_merge = cv2.vconcat(merged_masks)
            th = get_threshold(image = img_merge, thresh_option = thresh_method)
            print(f"Pixel intensity threshold for {ch} channel: {round(th, 2)}")

            positive2marker_dict = dct4positive2marker(df = df)

            if show_plot:

                n1 = random.randint(0,255)
                n2 = random.randint(0,255)
                n3 = random.randint(0,255)

                fig = go.Figure(data=[go.Histogram(x = list(img_merge[masks_merge != 0]), histnorm = 'probability', marker_color = f'rgb({n1},{n2},{n3})',
                                                   name = f"Px Intensity {ch}", showlegend = False)])
                fig.update_layout(title = f"Px Intensity {ch}",
                                  xaxis = dict(title = f"Px Intensity {ch}"),
                                  width = 1000,
                                  height = 400,
                                  template = "plotly_white",
                                  shapes= [{'line': {'color': 'red',
                                                     'dash': 'solid',
                                                     'width': 5},
                                            'type': 'line',
                                            'x0': th,
                                            'x1': th,
                                            'xref': 'x',
                                            'y0': -0.1,
                                            'y1': 1,
                                            'yref': 'paper'
                                           }
                                          ]
                                 )
                fig.show()

            for file in set(df["imageID"]):
                df_subset = df[df["imageID"] == file]
                mask = self.metadata["files"][file]["masks"]
                img = self.metadata["files"][file]["working_array"][self.metadata["channel_info"][ch]]
                thresh_img = custom_th_img(image = img, th = th)
                bin_img = binary_img(thresh_img)

                for index, row in df_subset.iterrows():
                    avg_binary = np.average(bin_img[mask == row["cellID"]])
                    if avg_binary >= frac_covered:
                        positive2marker_dict[row["imageID"]][row["cellID"]] = True
                    else:
                        positive2marker_dict[row["imageID"]][row["cellID"]] = False

            positive2marker_list = []
            for index, row in df.iterrows():
                positive2marker_list.append(positive2marker_dict[row["imageID"]][row["cellID"]])

            df[f"{ch}_positive_popStat"] = positive2marker_list


    def positive2marker_avgMethod(self, df = None, method = "triangle", show_plot = True, countOutliers = True):
        """
        Identifies cells positive to a marker by using average intensities and thresholding methods.

        Parameters
        ----------
        df : pandas DataFrame, optional
            NG generated DataFrame. The default is None.
        method : string, optional
            Thresholding method. The default is "triangle".

        Raises
        ------
        ValueError
            Error if thresholding method is not supported.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        lst_methods = ["otsu", "triangle", "adaptive_otsu", "adaptive_triangle"]

        if not method.lower() in lst_methods:
            raise ValueError(f"Method {method} is not supported.")

        for ch in self.metadata["channel_info"].keys():

            if ch == self.dna_marker:
                continue

            n1 = random.randint(0,255)
            n2 = random.randint(0,255)
            n3 = random.randint(0,255)

            vals = df[f"avg_intensity_{ch}"].copy()
            if not countOutliers:
                lower_th, higher_th = removeOutliers_keepSuspected(vals)
                vals = [v for v in df[f"avg_intensity_{ch}"].copy() if v >= lower_th and v <= higher_th]

            if method.lower() == "otsu":
                thresh = threshold_otsu(image = np.array(vals))
            elif method.lower() == "triangle":
                thresh = threshold_triangle(image = np.array(vals))
            elif method.lower() == "adaptive_otsu":
                fltd = gaussian(np.array(vals), 3)
                th = threshold_otsu(image = fltd)
                x1 = sorted(fltd)[0]
                x2 = sorted(fltd)[-1]
                y1 = sorted(vals)[0]
                y2 = sorted(vals)[-1]
                thresh = lin_int(x1, y1, x2, y2, th)
            elif method.lower() == "adaptive_triangle":
                fltd = gaussian(np.array(vals), 3)
                th = threshold_triangle(image = fltd)
                x1 = sorted(fltd)[0]
                x2 = sorted(fltd)[-1]
                y1 = sorted(vals)[0]
                y2 = sorted(vals)[-1]
                thresh = lin_int(x1, y1, x2, y2, th)

            if show_plot:
                fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)
                fig.add_trace(go.Box(x = df[f"avg_intensity_{ch}"], boxpoints = 'suspectedoutliers', fillcolor = f'rgba({n1},{n2},{n3},0.5)',
                                     line_color = f'rgb({n1},{n2},{n3})', showlegend = False, name = ''), row = 1, col = 1)
                fig.add_trace(go.Histogram(x = df[f"avg_intensity_{ch}"], histnorm = 'probability', marker_color = f'rgb({n1},{n2},{n3})',
                                          name = f"Average Intensity {ch}", showlegend = False, xaxis = 'x2'), row = 2, col = 1)
                fig.update_layout(title = f"Average Intensity {ch}",
                                  xaxis = dict(autorange = True, showgrid = True, zeroline = True, gridwidth=1),
                                  xaxis2 = dict(title = f"Average Intensity {ch}"),
                                  width = 1000,
                                  height = 400,
                                  template = "plotly_white",
                                  shapes= [{'line': {'color': 'red',
                                                     'dash': 'solid',
                                                     'width': 5},
                                            'type': 'line',
                                            'x0': thresh,
                                            'x1': thresh,
                                            'xref': 'x2',
                                            'y0': -0.1,
                                            'y1': 1,
                                            'yref': 'paper'
                                           }
                                          ]
                                 )

                fig.show()

            df[f"{ch}_positive_avgMethod"] = [True if v >= thresh else False for v in df[f"avg_intensity_{ch}"]]
            print(f"Threshold for Average Intensity {ch}: {round(thresh)} AU... {len(df[df['avg_intensity_' + ch] >= thresh])} positive cells in DataSet.")


    def population_intensityNorm(self, df = None):
        """
        Normalises intensities

        Parameters
        ----------
        df : pandas DataFrame, optional
            NG generated DataFrame. The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        for col in tqdm(df.columns):
            if "intensity" in col:
                if not "normalised" in col:
                    temp_lst = np.array(roundValues(df[col].to_list()))+1
                    mode_ = statistics.mode(temp_lst)
                    df[f"{col}_normalised"] = np.array(temp_lst)/mode_



    def selection2df(self, table, lr_corr = False, original_df = None, feature1 = None, feature2 = None):
        """
        Generates pandas dataframe from selection table.

        Parameters
        ----------
        table : Plotly Table
            Nuclear Features Data as Plotly Table.

        Returns
        -------
        df_out : pandas dataframe
            DataFrame from selection.

        """
        d = table.to_dict()
        df_out = pd.DataFrame(d['data'][0]['cells']['values'], index = d['data'][0]['header']['values']).T
        df_out = df_out.reset_index(drop = True)
        print(f"A total of {len(df_out)} cells on selection.", end = " ")
        for ch in self.metadata["channel_info"].keys():
            if ch == self.dna_marker:
                continue
            try:
                print(f"{len(df_out[df_out[ch + '_positive_avgMethod'] == True])} {ch}-positive cells.", end = " ")
            except:
                print(f"{len(df_out[df_out[ch + '_positive'] == True])} {ch}-positive cells.", end = " ")

        if lr_corr:

            if not isinstance(original_df, pd.DataFrame):
                raise ValueError("Please provide us with the DataFrame used for ScatterWidget")
            if feature1 == None:
                raise ValueError("Please provide us with the 'feature1' used for ScatterWidget")
            if feature2 == None:
                raise ValueError("Please provide us with the 'feature2' used for ScatterWidget")

            df_out = linearReg_correction(sel_df = df_out, df = original_df, x = feature2, y = feature1)
            df_out = df_out.reset_index(drop = True)

            print(f"A total of {len(df_out)} cells after linear regression correction.", end = " ")
            for ch in self.metadata["channel_info"].keys():
                if ch == self.dna_marker:
                    continue
                try:
                    print(f"{len(df_out[df_out[ch + '_positive_avgMethod'] == True])} {ch}-positive cells.", end = " ")
                except:
                    print(f"{len(df_out[df_out[ch + '_positive'] == True])} {ch}-positive cells.", end = " ")

        return df_out


    def scatter_widget(self, feature1, feature2, df = None, xlog = False, ylog = False, size = 20, subgroup = None):
        """
        Generates Scatter Widget for data selection

        Parameters
        ----------
        feature1 : str
            Nuclear feature 1.
        feature2 : str
            Nuclear feature 2.
        xlog : bool, optional
            True for X axis to be in log scale. The default is False.
        ylog : TYPE, optional
            True for Y axis to be in log scale. The default is False.

        Returns
        -------
        f : Figure Widget
            Scatterplot Widget of selescted nuclear features.

        """

        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        df = df.copy()

        if subgroup is None:
            f = go.FigureWidget([go.Scatter(y = df[feature1],
                                            x = df[feature2],
                                            mode = 'markers',
                                            marker=dict(size = size
                                                        )
                                            )
                                 ]
                                )
        else:
            f = go.FigureWidget([go.Scatter(y = df[feature1],
                                            x = df[feature2],
                                            mode = 'markers',
                                            marker=dict(size = size,
                                                        color = ["red" if row[subgroup] == True else "gray"
                                                                 for index, row in df.iterrows()]
                                                        )
                                            )
                                 ]
                                )

        if xlog == False and ylog == False:

            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2),
                yaxis = dict(title = feature1),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )

        elif xlog == True and ylog == False:

            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2, type = "log"),
                yaxis = dict(title = feature1),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )

        elif xlog == False and ylog == True:

            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2),
                yaxis = dict(title = feature1, type = "log"),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )

        elif xlog == True and ylog == True:

            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2, type = "log"),
                yaxis = dict(title = feature1, type = "log"),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )

        scatter = f.data[0]

        t = go.FigureWidget([go.Table(
        header=dict(values = df.columns,
                    fill = dict(color='#C2D4FF'),
                    align = ['left'] * 5),
        cells=dict(values=[df[col] for col in df.columns],
                   fill = dict(color='#F5F8FF'),
                   align = ['left'] * 5))])

        def selection_fn(trace,points,selector):
            t.data[0].cells.values = [df.reindex(index = points.point_inds)[col] for col in df.columns]

        scatter.on_selection(selection_fn)

        return f, t


    def print_features(self, df = None):
        """
        Prints measured nuclear features.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        lst_features = get_lst_features(df = df)

        for ft in lst_features:
            print(ft)


    def plot_boxplot_hist(self, feature = "nuclear_area", df = None):
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

        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        ft2show = df[feature].to_list()

        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)
        fig.add_trace(go.Box(x = ft2show, boxpoints = 'suspectedoutliers', fillcolor = 'rgba(7,40,89,0.5)',
                             line_color = 'rgb(7,40,89)', showlegend = False, name = ''), row = 1, col = 1)
        fig.add_trace(go.Histogram(x = ft2show, histnorm = 'probability', marker_color = 'rgb(7,40,89)',
                                  name = feature, showlegend = False), row = 2, col = 1)
        fig.update_layout(title = f"{feature} distribution",
                          xaxis = dict(autorange = True, showgrid = True, zeroline = True, gridwidth=1), width = 1000,
                          height = 400, template = "plotly_white")

        return fig


    def filter_data(self, feature, min_value = None, max_value = None, df = None):
        """
        Filter data with lower and higher thresholds.

        Parameters
        ----------
        feature : str
            Nuclear feature.
        min_value : float, optional
            Minimum value required for the given nuclear feature. The default is None.
        max_value : TYPE, optional
            Maximum value required for the given nuclear feature. The default is None.
        df : pandas dataframe, optional
            DataFrame containing the nuclear features. The default is None.

        Returns
        -------
        df_out : pandas dataframe
            Filtered DataFrame containing the nuclear features.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        before = len(df)

        if min_value != None and max_value != None:
            df_out = df[(df[feature] >= min_value) & (df[feature] <= max_value)]
        elif min_value == None and max_value != None:
            df_out = df[(df[feature] <= max_value)]
        elif min_value != None and max_value == None:
            df_out = df[(df[feature] >= min_value)]
        elif min_value == None and max_value == None:
            df_out = df

        after = len(df_out)
        df_out = df_out.reset_index(drop = True)

        print(f"{before - after} cells were removed with the filter. DataSet is now conformed by {len(df_out)}.")

        return df_out


    def spearman_cc(self, feature, df = None):
        """
        Determine Spearman's correlation coefficients between nuclear features.

        Parameters
        ----------
        feature : string
            Nuclear feature to find Spearman's cc.
        df : pandas DataFrame, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        ### --- Create a copy of Data DataFrame for removing NAN values
        data_notnan = df.copy()

        ### --- Remove cells with NAN features
        for key in data_notnan.columns:
            data_notnan = data_notnan[data_notnan[key].notna()]

        dct_corr = {}

        for column in data_notnan.columns:
            if not column in [feature, "cellID", "imageID", "x_pos", "y_pos"]:
                corr, _ = stats.spearmanr(data_notnan[feature], data_notnan[column])
                dct_corr[column] = round(corr, 3)

        SORT = sorted(dct_corr.items(), key = operator.itemgetter(1), reverse = True)

        print(f"Spearman's correlation coefficients for {feature} and:\n")
        for n, group in enumerate(SORT):
            print(f"\t{n+1}.- {group[0]}: {group[1]}")


    def plot_spearman(self, df = None):
        """
        Plots Spearman's correlation coefficients for all features.

        Parameters
        ----------
        df : pandas DataFrame, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        df = df[[col for col in df.columns if not col in ["cellID", "imageID", "x_pos", "y_pos"] and not "normalised" in col]]

        corr = df.corr(method = 'spearman')

        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(17,15))
        sns.heatmap(corr, ax = ax, center = 0, cmap = cmap, mask = mask,
                    cbar_kws={"shrink": .5}, linewidths=.5)

        return fig


    def show_cell(self, df, order_by = "nuclear_area", fig_height = 15, fig_width = 40, show_nucleus = True,
                  contrast_red = 3, contrast_green = 3, contrast_blue = 4, uniqID = False, scatterMarker_selected = False):
        """
        Shows selected cells from DataFrame.

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        order_by : TYPE, optional
            DESCRIPTION. The default is "nuclear_area".
        fig_height : TYPE, optional
            DESCRIPTION. The default is 15.
        fig_width : TYPE, optional
            DESCRIPTION. The default is 40.
        show_nucleus : TYPE, optional
            DESCRIPTION. The default is True.
        contrast_red : TYPE, optional
            DESCRIPTION. The default is 3.
        contrast_green : TYPE, optional
            DESCRIPTION. The default is 3.
        contrast_blue : TYPE, optional
            DESCRIPTION. The default is 4.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        df = df.copy()
        # Ask for the number of cells to show
        while True:
            no_cells = input('\nEnter number of nuclei to show (any integer OR "all"): ')
            try:
                no_cells = int(no_cells)
                break
            except:
                if isinstance(no_cells, str):
                    if no_cells.lower() == 'all':
                        no_cells = len(df)
                        break
                else:
                    print('Ops! Invalid number format! Enter an integer or "all"')


        if len(df) == no_cells:
            print(f"\nShowing all cells ({len(df)}) in the selected area")

        if len(df) > no_cells:
            print('\nShowing {0} cells of a total of {1} in the selected data'.format(no_cells, str(len(df))))

        if len(df) < no_cells:
            no_cells = len(df)
            print('\nONLY ' + str(len(df)) + ' cells were found in the selected data')

        new_df = df.sample(n = no_cells)

        # Get the names of the channels
        lst_channels = [mkr for mkr in self.metadata["channel_info"].keys() if mkr != self.dna_marker]
        dct_channels = {}

        if not scatterMarker_selected:
            for ch in lst_channels:
                dct_channels[ch] = input(f'Show {ch} (y/n): ')
        else:
            try:
                dct_channels[self.scatter_mkr] = "y"
            except:
                print("Ops! Should run 'scatterMarker' function before.")

        # Color of the channels
        dct_colors = {}

        for ch in dct_channels:
            if dct_channels[ch].lower() == 'y':
                while True:
                    try:
                        dct_colors[ch] = input('Desired colour for {0} (red/green/blue): '.format(ch))
                        if dct_colors[ch] == 'red' or dct_colors[ch] == 'green' or dct_colors[ch] == 'blue':
                            break
                        else:
                            raise ValueError
                    except:
                        print('Input color {0} is not valid!'.format(dct_colors[ch]))
                        pass

        # Generate the figure
        if no_cells <= 5:
            fig, axes = plt.subplots(nrows = no_cells, ncols = 1, sharex = True, sharey = True,
                                     figsize = (int(fig_width)/2.54, (int(fig_height)/2.54) * no_cells))
        elif no_cells > 5 and no_cells <= 10:
            fig, axes = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True,
                                    figsize = (int(fig_width)/2.54, (int(fig_height)/2.54)))
        elif no_cells > 10:
            if int(str(no_cells)[-1]) > 5:
                fig, axes = plt.subplots(nrows = ((no_cells // 5) + 1), ncols = 5, sharex = True, sharey = True,
                                        figsize = (int(fig_width)/2.54, (int(fig_height)/2.54) * ((no_cells // 10) + 1)))
            elif int(str(no_cells)[-1]) <= 5 and int(str(no_cells)[-1]) > 0:
                fig, axes = plt.subplots(nrows = (no_cells // 5 + 1), ncols = 5, sharex = True, sharey = True,
                                        figsize = (int(fig_width)/2.54, (int(fig_height)/2.54) * ((no_cells // 10) + 0.5)))
            elif int(str(no_cells)[-1]) == 0:
                fig, axes = plt.subplots(nrows = (no_cells // 5), ncols = 5, sharex = True, sharey = True,
                                        figsize = (int(fig_width)/2.54, (int(fig_height)/2.54) * (no_cells // 10)))

        ax = axes.ravel()

        # Generate ordered DataFrame
        ordered_df = new_df.sort_values(by = order_by, ascending = True)

        # Create Unique ID dictionary
        if uniqID:
            uniqID_dct = {}

        # Generate the figure
        print(f"\nGenerating figure in ascending order for the feature '{order_by}'...")
        n = 0
        for index, row in tqdm(ordered_df.iterrows(), total = ordered_df.shape[0]):
            masks = self.metadata["files"][row["imageID"]]["masks"].copy()
            wk_array = self.metadata["files"][row["imageID"]]["working_array"].copy()
            nucleus = wk_array[self.metadata["channel_info"][self.dna_marker]].copy()
            nucleus[masks != row['cellID']] = 0
            cY = int(row['y_pos'])
            cX = int(row['x_pos'])
            cY_low = cY - 150
            cY_high = cY + 150
            cX_low = cX - 150
            cX_high = cX + 150
            if (cY-150) < 0:
                cY_low = 0
            if (cY+150) > len(nucleus):
                cY_high = len(nucleus)
            if (cX-150) < 0:
                cX_low = 0
            if (cX+150) > len(nucleus[0]):
                cX_high = len(nucleus[0])
            nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
            y, x = nucleus.shape
            color_red = Image.fromarray(np.zeros((y, x, 3), dtype = 'uint8')).convert('L')
            color_green = Image.fromarray(np.zeros((y, x, 3), dtype = 'uint8')).convert('L')
            color_blue = Image.fromarray(np.zeros((y, x, 3), dtype = 'uint8')).convert('L')
            for ch in dct_channels:
                if dct_channels[ch].lower() == 'y':
                    channel = wk_array[self.metadata["channel_info"][ch]].copy()
                    channel = channel[cY_low:cY_high, cX_low:cX_high]
                    channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    RGB = np.array((*"RGB",))
                    if dct_colors[ch].lower() == 'red':
                        color = np.multiply.outer(channel, RGB == 'R')
                        color_red = Image.fromarray(color).convert('L')
                        enhancer = ImageEnhance.Contrast(color_red)
                        color_red = enhancer.enhance(contrast_red)
                    elif dct_colors[ch].lower() == 'green':
                        color = np.multiply.outer(channel, RGB == 'G')
                        color_green = Image.fromarray(color).convert('L')
                        enhancer = ImageEnhance.Contrast(color_green)
                        color_green = enhancer.enhance(contrast_green)
                    elif dct_colors[ch].lower() == 'blue':
                        color = np.multiply.outer(channel, RGB == 'B')
                        color_blue = Image.fromarray(color).convert('L')
                        enhancer = ImageEnhance.Contrast(color_blue)
                        color_blue = enhancer.enhance(contrast_blue)
            mrg = Image.merge("RGB", (color_red, color_green, color_blue))
            mrg = np.array(mrg, dtype = 'uint8')
            if show_nucleus == False:
                kernel = np.ones((3, 3), np.uint8)
                masks[masks != row['cellID']] = 0
                masks[masks == row['cellID']] = 1
                masks = masks[cY_low:cY_high, cX_low:cX_high]
                masks = np.uint16(masks)
                eroded = cv2.erode(masks, kernel, iterations = 2)
                nucleus = masks - eroded
                nucleus = np.array(nucleus, dtype = 'uint8')
                mrg[nucleus == 1] = [255, 255, 255]
                ax[n].set_xlim(0,299)
                ax[n].set_ylim(299,0)
                ax[n].imshow(mrg)
            if show_nucleus == True:
                nucleus = np.ma.masked_where(nucleus == 0, nucleus)
                ax[n].set_xlim(0,299)
                ax[n].set_ylim(299,0)
                ax[n].imshow(mrg)
                ax[n].imshow(nucleus)
            if uniqID:
                ax[n].set_title(f"ID: {n+1}", fontdict = {'fontsize' : 8})
                uniqID_dct[str(n+1)] = {"cellID": row["cellID"], "imageID": row["imageID"]}
            else:
                ax[n].set_title(f"{row['imageID']} | {row['cellID']}", fontdict = {'fontsize' : 8})
            ax[n].axis("off")
            n += 1
        plt.tight_layout()

        if uniqID:
            return fig, uniqID_dct
        else:
            return fig


    def scatter_dimReduction(self, features, method = "umap", df = None, feature4cmap = None, feature4cmap_scale = None, show_markers = False, random_state = False, scale = None, size = 10):
        """
        Perfoms dimension reduction and plots it on a widget.

        Parameters
        ----------
        features : list
            list of features to be used for dimension reduction.
        method : string, optional
            method to be employed for dimension reduction. The default is "umap".
        df : pandas DataFrame, optional
            DataFrame used as imageID. The default is None.
        feature4cmap : string, optional
            feature to be used for colormap. The default is None.
        show_markers : boolean, optional
            Show markers as colors in Scatter. The defailt is False.

        Raises
        ------
        ValueError
            Raises error if input is not correct.

        Returns
        -------
        f : figure widget
            figure showing dimension reduction of data.
        t : table widget
            table containing data selected.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        data = df.copy()

        corrFts, l = check_features(data, features)
        if not corrFts:
            raise ValueError(f"The feature '{l}' does not exist.")

        if not isinstance(features, list):
            raise ValueError("The parameter 'features' must be a list.")

        if show_markers:

            if feature4cmap != None:
                print(f"Will ignore the feature '{feature4cmap}' colormap as scatter color will be used for markers. Set 'show_markers' to False to show feature colormap instead of markers.")

            lst_markers = [mkr for mkr in self.metadata["channel_info"].keys() if mkr != self.dna_marker]

            if method.lower() == "pca":

                print("\nPlotting PCA...")
                f, t = plot_pca(df = data, features = features, show_markers = show_markers, lst_markers = lst_markers, scale = scale, size = size)

            elif method.lower() == "tsne":

                print("\nPlotting tSNE...")
                f, t = plot_tsne(df = data, features = features, show_markers = show_markers, lst_markers = lst_markers, scale = scale, size = size)

            elif method.lower() == "umap":

                print("\nPlotting UMAP...")
                f, t = plot_umap(df = data, features = features, show_markers = show_markers, lst_markers = lst_markers, random_state = random_state, scale = scale, size = size)

        else:

            if feature4cmap == None:
                raise ValueError("Please provide 'feature4cmap'.")

            print(f"Will perform {method} dimension reduction using {feature4cmap} as colormap.")

            if method.lower() == "pca":

                print("\nPlotting PCA...")
                f, t = plot_pca(df = data, features = features, feature4cmap = feature4cmap, scale = scale, size = size)

            elif method.lower() == "tsne":

                print("\nPlotting tSNE...")
                f, t = plot_tsne(df = data, features = features, feature4cmap = feature4cmap, scale = scale, size = size)

            elif method.lower() == "umap":

                print("\nPlotting UMAP...")
                f, t = plot_umap(df = data, features = features, feature4cmap = feature4cmap, scale = scale, random_state = random_state, size = size)

        return f, t


    def dist_plot(self, df = None, feature = "nuclear_area", split = False, bin_size = "auto", show_hist = True):
        """
        Generates distribution plot.

        Parameters
        ----------
        df : pandas DataFrame, optional
            NG generated DataFrame. The default is None.
        feature : string, optional
            Nuclear feature to plot. The default is "nuclear_area".
        split : boolean, optional
            Split plot by channels. The default is False.
        bin_size : int OR string, optional
            Bin size for histogram. The default is "auto".
        show_hist : boolean, optional
            Show histogram. The default is True.

        Raises
        ------
        ValueError
            Raises error if bin size is not an integer or 'auto'.

        Returns
        -------
        None.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        data = df.copy()

        hist_data = []
        group_labels = []
        if split:
            for ch in self.metadata["channel_info"].keys():
                if ch == self.dna_marker:
                    continue
                try:
                    hist_data.append(data[feature][data[f"{ch}_positive_avgMethod"] == True])
                except:
                    hist_data.append(data[feature][data[f"{ch}_positive"] == True])
                group_labels.append(ch)
        else:
            hist_data.append(data[feature])
            group_labels.append("all cells")

        if bin_size != "auto":
            try:
                bin_size = int(bin_size)
            except:
                raise ValueError("Bin size must be an integer or 'auto'!")

        fig = ff.create_distplot(hist_data, group_labels, bin_size = bin_size, show_hist = show_hist)
        fig.update_layout(template = 'plotly_white', title = feature)

        fig.show()


    def scatterMarker(self, df = None):
        """
        Show scatter for selecting cells to verify if they are positive to a marker.

        Parameters
        ----------
        df : pandas DataFrame, optional
            NG generated DataFrame. The default is None.

        Returns
        -------
        f : figure
            Scatter Widget.
        t : table
            Table Widget.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        while True:
            self.scatter_mkr = input(f"Enter a marker ({'/'.join([key for key in self.metadata['channel_info'].keys() if key != self.dna_marker])}): ")
            if self.scatter_mkr in list(self.metadata["channel_info"].keys()):
                break
            else:
                print(f"Ops! {self.scatter_mkr} is not a valid marker... Try again.")

        try:
            f, t = self.scatter_widget(feature1 = f'{self.scatter_mkr}_positive_avgMethod', # Plotted on y axis
                                       feature2 = f'avg_intensity_{self.scatter_mkr}', # Plotted on x axis
                                       df = df, # If None is given, then it uses raw data (nga.df_raw).
                                       xlog = True, # Use log scale for x axis (True = yes, False = no)
                                       ylog = False # Use log scale for y axis (True = yes, False = no)
                                       )
            return f, t
        except:
            try:
                f, t = self.scatter_widget(feature1 = f'{self.scatter_mkr}_positive', # Plotted on y axis
                                           feature2 = f'avg_intensity_{self.scatter_mkr}', # Plotted on x axis
                                           df = df, # If None is given, then it uses raw data (nga.df_raw).
                                           xlog = True, # Use log scale for x axis (True = yes, False = no)
                                           ylog = False # Use log scale for y axis (True = yes, False = no)
                                           )
                return f, t
            except:
                print("Please run a function for identifying cells positive to markers!")


    def assignAsPositive(self, df, uniqID_dct):
        """
        Assigns cells as positive to a marker

        Parameters
        ----------
        df : pandas DataFrame
            NG generated DataFrame.
        uniqID_dct : dictionary
            Cell ID dictionary.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print(f"Enter cell IDs as integer alone (e.g. 2) or range (e.g. 2-4) to assign cell as positive to {self.scatter_mkr}. Enter 0 to stop.\n")

        uniqID_dct = indexConversion(df = df, uniqID_dct = uniqID_dct)
        try:
            before = len(df[df[f"{self.scatter_mkr}_positive_avgMethod"] == True])
        except:
            before = len(df[df[f"{self.scatter_mkr}_positive"] == True])

        while True:
            ids = input(f"Enter cell IDs to mark as positive to {self.scatter_mkr}: ")
            ids = ids.replace(" ", "")
            if "-" in ids:
                ids = ids.split("-")
                if len(ids) > 2:
                    raise ValueError("Invalid format for range of IDs!")
                beg = int(ids[0])
                end = int(ids[1]) + 1
                for n in range(beg, end):
                    try:
                        df.at[uniqID_dct[str(n)], f'{self.scatter_mkr}_positive_avgMethod'] = True
                    except:
                        df.at[uniqID_dct[str(n)], f'{self.scatter_mkr}_positive'] = True
            elif ids == "0":
                break
            else:
                try:
                    int(ids)
                except:
                    raise ValueError("Invalid ID format!")
                try:
                    df.at[uniqID_dct[ids], f'{self.scatter_mkr}_positive_avgMethod'] = True
                except:
                    df.at[uniqID_dct[ids], f'{self.scatter_mkr}_positive'] = True

        try:
            after = len(df[df[f"{self.scatter_mkr}_positive_avgMethod"] == True])
        except:
            after = len(df[df[f"{self.scatter_mkr}_positive"] == True])

        print(f"\nBefore correction: {before} cells marked as positive to {self.scatter_mkr}.")
        print(f"After correction: {after} cells marked as positive to {self.scatter_mkr}.")


    def assignAsNegative(self, df, uniqID_dct):
        """
        Assigns cells as negative to a marker

        Parameters
        ----------
        df : pandas DataFrame
            NG generated DataFrame.
        uniqID_dct : dictionary
            Cell ID dictionary.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print(f"Enter cell IDs as integer alone (e.g. 2) or range (e.g. 2-4) to assign cell as negative to {self.scatter_mkr}. Enter 0 to stop.\n")

        uniqID_dct = indexConversion(df = df, uniqID_dct = uniqID_dct)
        try:
            before = len(df[df[f"{self.scatter_mkr}_positive_avgMethod"] == False])
        except:
            before = len(df[df[f"{self.scatter_mkr}_positive"] == False])

        while True:
            ids = input(f"Enter cell IDs to mark as negative to {self.scatter_mkr}: ")
            ids = ids.replace(" ", "")
            if "-" in ids:
                ids = ids.split("-")
                if len(ids) > 2:
                    raise ValueError("Invalid format for range of IDs!")
                beg = int(ids[0])
                end = int(ids[1]) + 1
                for n in range(beg, end):
                    try:
                        df.at[uniqID_dct[str(n)], f'{self.scatter_mkr}_positive_avgMethod'] = False
                    except:
                        df.at[uniqID_dct[str(n)], f'{self.scatter_mkr}_positive'] = False
            elif ids == "0":
                break
            else:
                try:
                    int(ids)
                except:
                    raise ValueError("Invalid ID format!")
                try:
                    df.at[uniqID_dct[ids], f'{self.scatter_mkr}_positive_avgMethod'] = False
                except:
                    df.at[uniqID_dct[ids], f'{self.scatter_mkr}_positive'] = False

        try:
            after = len(df[df[f"{self.scatter_mkr}_positive_avgMethod"] == False])
        except:
            after = len(df[df[f"{self.scatter_mkr}_positive"] == False])

        print(f"\nBefore correction: {before} cells marked as negative to {self.scatter_mkr}.")
        print(f"After correction: {after} cells marked as negative to {self.scatter_mkr}.")


    def add2subgroup(self, df, uniqID_dct, subgroupName):
        """
        Adds selected cells to saved in uniqID dictionary to subgroup.

        Parameters
        ----------
        df : pandas DataFrame
            NG generated DataFrame.
        uniqID_dct : dict
            Unique ID dictionary.
        subgroupName : string
            Subgroup Name.

        Raises
        ------
        ValueError
            Raises error if invalid ID is entered.

        Returns
        -------
        None.

        """
        while True:
            filter_data = input("Filter selected data (y/n)? ")
            if filter_data.lower() in ["y", "n"]:
                break
            else:
                print("\nInvalid input! Please try again...")

        if filter_data == "y":

            print(f"\nEnter cell IDs as integer alone (e.g. 2) or range (e.g. 2-4) to remove from subgroup '{subgroupName}'. Enter 0 to stop.\n")

            while True:
                ids = input(f"Enter cell IDs to remove from '{subgroupName}': ")
                ids = ids.replace(" ", "")
                if "-" in ids:
                    ids = ids.split("-")
                    if len(ids) > 2:
                        raise ValueError("Invalid format for range of IDs!")
                    beg = int(ids[0])
                    end = int(ids[1]) + 1
                    for n in range(beg, end):
                        uniqID_dct.pop(str(n))
                elif ids == "0":
                    break
                else:
                    try:
                        int(ids)
                    except:
                        raise ValueError("Invalid ID format!")
                    uniqID_dct.pop(str(ids))

        uniqID_dct = indexConversion(df = df, uniqID_dct = uniqID_dct)

        pos2subgroup = []
        for index, row in df.iterrows():
            if index in list(uniqID_dct.values()):
                pos2subgroup.append(True)
            else:
                pos2subgroup.append(False)

        df[subgroupName] = pos2subgroup

        print(f"\nSubgroup '{subgroupName}' is conformed by {len(df[df[subgroupName] == True])} cells.")


    def countCellsCat(self, data, categories = None):
        """
        Counts the number of positive cells for each of the channels,
        except DNA staining marker

        Parameters
        ----------
        data : pandas DataFrame
            NG DataFrame containing at least a marker protein.
        categories : str or list, optional
            List of experiment identifiers to categorise data. The default is None.

        Raises
        ------
        ValueError
            Raises error if categories is not list or string.

        Returns
        -------
        df_count : pandas DataFrame
            DataFrame containing counts of cells positive to markers.

        """
        if not isinstance(categories, str) or not isinstance(categories, list):
            raise ValueError("Parameter 'categories' should be either a list or a string.")

        channels = list(self.metadata["channel_info"].keys())

        dct_count = {"general": {}}

        for ch in tqdm(channels):
            if ch == self.dna_marker:
                continue
            if f"{ch}_positive_avgMethod" in list(data.columns):
                dct_count["general"][f"{ch}+"] = 0
                for index, row in data.iterrows():
                    if row[f"{ch}_positive_avgMethod"]:
                        dct_count["general"][f"{ch}+"] += 1
                        if isinstance(categories, str):
                            if not categories in list(dct_count.keys()):
                                dct_count[categories] = {}
                            if not f"{ch}+" in dct_count[categories]:
                                dct_count[categories][f"{ch}+"] = 0
                            if categories in row["file"]:
                                dct_count[categories][f"{ch}+"] += 1
                        elif isinstance(categories, list):
                            for c in categories:
                                if not c in list(dct_count.keys()):
                                    dct_count[c] = {}
                                if not f"{ch}+" in dct_count[c]:
                                    dct_count[c][f"{ch}+"] = 0
                                if c in row["file"]:
                                    dct_count[c][f"{ch}+"] += 1

        df_count = pd.DataFrame.from_dict(data = dct_count)

        return df_count


    def countCells(self, data, byFile = False):
        """
        Counts the number of positive cells for each of the channels,
        except DNA staining marker

        Parameters
        ----------
        data : pandas DataFrame
            NG DataFrame containing at least a marker protein.
        byFile : boolean, optional
            Add information per file. The default is False.

        Returns
        -------
        df_count : pandas DataFrame
            DataFrame containing counts of cells positive to markers.

        """

        channels = list(self.metadata["channel_info"].keys())

        dct_count = {"general": {}}
        dct_count["general"]["total"] = len(data)

        for ch in tqdm(channels):

            if ch == self.dna_marker:
                continue

            if f"{ch}_positive_avgMethod" in list(data.columns):
                dct_count["general"][f"{ch}+"] = len(data[data[f"{ch}_positive_avgMethod" ] == True])

                if byFile:
                    for file in np.unique(data["file"]):
                        subset = data[data["file"] == file]
                        if not file in dct_count:
                            dct_count[file] = {}
                            dct_count[file]["total"] = len(subset)
                        dct_count[file][f"{ch}+"] = len(subset[subset[f"{ch}_positive_avgMethod" ] == True])

        # Double positives
        channels.remove(self.dna_marker)
        dct_count["general"][f"{channels[0]}+ & {channels[1]}+"] = len(data[(data[f"{channels[0]}_positive_avgMethod" ] == True) & (data[f"{channels[1]}_positive_avgMethod" ] == True)])
        dct_count["general"][f"{channels[1]}+ & {channels[2]}+"] = len(data[(data[f"{channels[1]}_positive_avgMethod" ] == True) & (data[f"{channels[2]}_positive_avgMethod" ] == True)])
        dct_count["general"][f"{channels[0]}+ & {channels[2]}+"] = len(data[(data[f"{channels[0]}_positive_avgMethod" ] == True) & (data[f"{channels[2]}_positive_avgMethod" ] == True)])

        if byFile:
            for file in np.unique(data["file"]):
                subset = data[data["file"] == file]
                dct_count[file][f"{channels[0]}+ & {channels[1]}+"] = len(subset[(subset[f"{channels[0]}_positive_avgMethod" ] == True) & (subset[f"{channels[1]}_positive_avgMethod" ] == True)])
                dct_count[file][f"{channels[1]}+ & {channels[2]}+"] = len(subset[(subset[f"{channels[1]}_positive_avgMethod" ] == True) & (subset[f"{channels[2]}_positive_avgMethod" ] == True)])
                dct_count[file][f"{channels[0]}+ & {channels[2]}+"] = len(subset[(subset[f"{channels[0]}_positive_avgMethod" ] == True) & (subset[f"{channels[2]}_positive_avgMethod" ] == True)])


        df_count = pd.DataFrame.from_dict(data = dct_count)

        return df_count


    def plot_violinPlots(self, data, features = "all", byFile = False, scale = "area", sharex = True):
        """
        Plots violinplots for input features.

        Parameters
        ----------
        data : pandas DataFrame
            NG generated DataFrame containing measured features.
        features : str or list, optional
            Features to plot. The default is "all".
        byFile : boolean, optional
            Plot figures by file. The default is False.
        scale : str, optional
            {“area”, “count”, “width”}. The default is "area".

        Raises
        ------
        ValueError
            Raises error if features parameter is not list or 'all'.

        Returns
        -------
        fig : matplotlib figure
            Violin plots figure showing the features given as input.

        """
        if not isinstance(features, list) and features != "all":
            raise ValueError("Parameter 'features' is not valid. Should be string or list.")

        if features == "all":
            features = [f for f in data.columns if not f in ["imageID", "file", "cellID"] and not "positive" in f]
        elif isinstance(features, list):
            features_ = []
            for f in features:
                if f in list(data.columns):
                    features_.append(f)
                else:
                    print(f"Feature '{f}' will be omitted.")
            features = features_

        n_fts = len(features)

        if byFile:
            cols = len(np.unique(data["file"]))
        else:
            cols = 1

        fig, axes = plt.subplots(nrows = n_fts, figsize = (4*cols, 4*n_fts), sharex = sharex)

        ax = axes.ravel()

        temp = data.copy()


        for n, ft in enumerate(features):

            temp[ft] = temp[ft].astype(float)

            if byFile:
                sns.violinplot(x = "file",
                               y = ft,
                               data = temp,
                               palette = "Set3",
                               scale = scale,
                               bw = .2,
                               ax = ax[n]
                               )
            else:
                sns.violinplot(y = ft,
                               data = temp,
                               palette = "Set3",
                               scale = scale,
                               bw = .2,
                               ax = ax[n]
                               )

        return fig


    def plot_radar(self, df, features, subgroup = None, mode = "mean"):
        """
        Plots radar plot with mean or median values.

        Parameters
        ----------
        df : pandas DataFrame
            NG-generated DataFrame.
        features : list
            List of features to plot.
        subgroup : str or None, optional
            Name of subgroup to plot. The default is None.
        mode : str, optional
            Choose between 'mean' and 'median'. The default is "mean".

        Returns
        -------
        fig : figure axes
            Returns radar plot.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw

        data = df.copy()
        data = data[features]

        data = pd.DataFrame(MinMaxScaler().fit_transform(data.values), columns = data.columns)

        if subgroup != None:
            data = data[df[subgroup] == True]

        if mode == "mean":
            data = data.mean()
        elif mode == "median":
            data = data.median()

        data = data.T.reset_index()
        data = data.rename(columns={"index": "theta", 0: "r"})

        fig = px.line_polar(data, r='r', theta='theta', line_close=True, range_r = [0,1])
        fig.update_traces(fill='toself')

        return fig


    def toCSV(self, df, filename = "filtered_output.csv"):
        """
        Export data generated as CSV

        Parameters
        ----------
        filename : str, optional
            Name of output file. The default is "filtered_output.csv".

        Returns
        -------
        None.

        """
        if not filename.endswith(".csv"):
            filename = filename + ".csv"

        df.to_csv(self.path_dir + filename, index = False)
        print(f"CSV file saved as: {self.path_dir + filename}")



###########################################
#     Functions & Classes | Clustering    #
###########################################



def calculate_wcss(data, show_plot = False):
    """
    Calculate Sum Squared Distances

    Parameters
    ----------
    data : pandas dataframe values
        Nuclear Features values.

    Returns
    -------
    wcss : lst
        list of kmeans inertia.

    """
    wcss = []
    K = range(2, 21)
    for n in K:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    if show_plot == True:
        plt.plot(K, wcss, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    return wcss


def optimal_number_of_clusters(wcss):
    """
    Find optimal number of clusters with Kmeans

    Parameters
    ----------
    wcss : lst
        list of kmeans inertia.

    Returns
    -------
    int
        Optimal number of clusters.

    """
    x1, y1 = 2, wcss[0]
    x2, y2 = 21, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


class NuclearGame_Clustering(object):

    def __init__(self, arg):
        """
        Start NuclearGame - Clustering

        Parameters
        ----------
        arg : str
            Path to CSV file containing nuclear features.

        Returns
        -------
        None.

        """
        self.path = arg
        self.data = pd.read_csv(self.path)


    def scale_data(self, features, method = "StandardScaler"):
        """
        Scale data

        Parameters
        ----------
        features : list
            List containing nuclear features to be considered for dimension reduction.
        method : str, optional
            Method for scaling (StandardScaler/MinMaxScaler/MaxAbsScaler/None).
            The default is "StandardScaler".

        Returns
        -------
        None.

        """

        self.features = features

        # Copy DataFrame
        self.data_notna = self.data.copy()

        # Remove cells that contain NAN values
        for ft in self.features:
            self.data_notna = self.data_notna[self.data_notna[ft].notna()]

        # Obtain relevant DataSet
        self.rel_data = self.data_notna.loc[:, self.features]

        # Obtain values of relevant DataSet
        self.values = self.rel_data.values

        # Scale data
        if method.lower() == "standardscaler":
            self.scaled_data = StandardScaler().fit_transform(self.values) # StandardScaler
        elif method.lower() == "minmaxscaler":
            self.scaled_data = MinMaxScaler().fit_transform(self.values) # MinMaxScaler
        elif method.lower() == "maxabsscaler":
            self.scaled_data = MaxAbsScaler().fit_transform(self.values) # MaxAbsScaler
        else:
            self.scaled_data = self.values


    def umap_reduction(self, show_plot = False, size = 10, n_neighbors = 15, min_dist = 0.1, n_components = 2):
        """
        Perform UMAP dimension reduction.

        Parameters
        ----------
        show_plot : bool, optional
            True for showing UMAP plot. The default is False.

        Returns
        -------
        None.

        """
        reducer = umap.UMAP(n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            n_components = n_components)

        embedding = reducer.fit_transform(self.scaled_data)

        self.principalDf = pd.DataFrame(data = embedding, columns = ['UMAP 1', 'UMAP 2'])

        if show_plot == True:
            fig, ax = plt.subplots()
            ax.scatter(self.principalDf["UMAP 1"], self.principalDf["UMAP 2"], s = size, cmap='Spectral')
            ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title("UMAP")
            plt.tight_layout()
            return fig



    def optimalClusters(self, show_plot = False):
        """
        Calculate optimal number of clusters

        Parameters
        ----------
        show_plot : bool, optional
            Show plot. The default is False.

        Returns
        -------
        None.

        """
        # Calculating the within clusters sum-of-squares for 20 cluster amounts
        sum_of_squares = calculate_wcss(self.scaled_data, show_plot)

        # Calculating the optimal number of clusters
        self.no_clusters = optimal_number_of_clusters(sum_of_squares)
        print(f"\nOptimal number of clusters: {self.no_clusters}")


    def clusterableEmbedding(self, n_neighbors = 50, min_dist = 0.0, n_components = 2):
        """
        Obtain a clusterable embedding

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors. The default is 50.
        min_dist : float, optional
            Minimum distance. The default is 0.0.
        n_components : int, optional
            Number of components. The default is 2.

        Returns
        -------
        None.

        """
        self.clusterable_embedding = umap.UMAP(
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            n_components = n_components,
            ).fit_transform(self.scaled_data)


    def kmeans_clustering(self, show_plot = False, size = 10):
        """
        Perform kmeans clustering

        Parameters
        ----------
        show_plot : bool, optional
            Show plot with clustering. The default is False.

        Returns
        -------
        None.

        """
        self.kmeans_labels = KMeans(n_clusters = self.no_clusters).fit_predict(self.clusterable_embedding)
        cdict = {0: 'grey', 1: 'red', 2: 'blue', 3: 'green', 4: 'pink', 5: 'orange', 6: 'yellow', 7: 'saddlebrown', 8: 'purple',
                9: 'magenta'}

        if show_plot == True:
            fig, ax = plt.subplots(figsize = (6.4, 4.3))
            for g in np.unique(self.kmeans_labels):
                ix = np.where(self.kmeans_labels == g)
                ax.scatter(np.array(self.principalDf["UMAP 1"])[ix], np.array(self.principalDf["UMAP 2"])[ix], c = cdict[g],
                           label = g, s = size)
            ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left'); ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title("kMeans Clustering")
            plt.tight_layout()
            return fig


    def assignCluster(self):
        """
        Assigns cluster number to cells.

        Returns
        -------
        pandas dataframe
            Nuclear features dataframe containing cluster numbers.

        """
        self.data_notna["cluster"] = [c for c in self.kmeans_labels]

        return self.data_notna


    def scatter_heatmap(self, feature, size = 10, cmap = "RdBu_r"):
        """
        Generate UMAP scatter plot heatmap.

        Parameters
        ----------
        feature : str
            Nuclear Feature to plot as Scatter plot heatmap.
        size : int, optional
            Circles size. The default is 10.
        cmap : Color map, optional
            Color map for heatmap. The default is "RdBu_r".

        Returns
        -------
        fig : matplot fig
            UMAP scatter plot heatmap for input feature.

        """
        fig, ax = plt.subplots()

        q3, q1 = np.percentile(self.data_notna[feature], [75 ,25])
        iqr = q3 - q1
        _max = q3 + (1.5 * iqr)
        _min = q1 - (1.5 * iqr)

        norm = plt.Normalize(_min, _max, clip = True)
        sca = ax.scatter(self.principalDf["UMAP 1"],
                         self.principalDf["UMAP 2"],
                         c = self.data_notna[feature],
                         s = size,
                         cmap = cmap,
                         norm = norm
                         )
        ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title(f"UMAP | {feature}")
        position = fig.add_axes([1.02, 0.80, 0.012, 0.15])
        fig.colorbar(sca, cax = position)
        #plt.tight_layout()

        return fig



