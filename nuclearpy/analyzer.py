# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:52:32 2021

@author: gabrielemilioherreraoropeza
"""
# TODO: Need to have more function checks

#############################################
#                 Imports                   #
#############################################
from math import ceil
import cv2
import anndata
import pandas as pd
import copy
import os, json
from os.path import exists, join, dirname
import statistics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import operator
from fnmatch import fnmatchcase

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3

#############################################
#     Functions & Classes | Segmentation    #
#############################################

def _log(txt, verbose=True):
    if verbose:
        print(txt)


def intensityNormalisation(df, method="mode", nbins=10, verbose=True, hue="experiment"):
    dct_norm = {}
    df_ = df.copy()
    method = method.lower()
    for n, exp in tqdm(enumerate(set(df_[hue])), total=len(set(df_[hue]))):
        subset = df_[df_[hue] == exp]
        dct_norm[exp] = {}
        for col in subset.columns:
            if "avg_intensity" in col and not any(l in col.lower() for l in ["dapi"]):
                if method == "mode":
                    relevant_subset = subset[[col]].copy()
                    relevant_subset["bins"] = pd.cut(relevant_subset[col], nbins, duplicates="drop",
                                                     labels=False)
                    bins_mode = statistics.mode(relevant_subset["bins"])
                    mode_ = relevant_subset[col][relevant_subset["bins"] == bins_mode].median()
                    subset[col] = subset[col] / mode_
                    dct_norm[exp][col] = mode_
                    _log(f"Reference value for {col} in {exp}: {mode_}", verbose)
                elif method == "mean":
                    mean_ = subset[col].mean()
                    subset[col] = subset[col] / mean_
                    dct_norm[exp][col] = mean_
                elif method == "median":
                    median_ = subset[col].median()
                    subset[col] = subset[col] / median_
                    dct_norm[exp][col] = median_
        if n == 0:
            out_df = subset.copy()
        else:
            out_df = out_df.append(subset)
    return out_df, dct_norm


def find_SingleCells(df, byExperiment=True, nbins=10, spread=0.2, channel="dapi", hue="experiment"):
    df_ = df.copy()
    dct_norm = {}
    col = f"total_intensity_{channel}"

    if not col in list(df_.columns):
        raise ValueError("Ops! Channel not found.")

    if byExperiment:
        for n, exp in tqdm(enumerate(set(df_[hue])), total=len(set(df_[hue]))):
            subset = df_[df_[hue] == exp]
            temp = subset.copy()
            temp["bins"] = pd.cut(temp[col], nbins, duplicates="drop", labels=False)
            bins_mode = statistics.mode(temp["bins"])
            mode_ = temp[col][temp["bins"] == bins_mode].median()
            subset[col] = subset[col] / mode_
            dct_norm[exp] = mode_
            if n == 0:
                out_df = subset.copy()
            else:
                out_df = out_df.append(subset)
    elif not byExperiment:
        temp = df_.copy()
        temp["bins"] = pd.cut(temp[col], nbins, duplicates="drop", labels=False)
        bins_mode = statistics.mode(temp["bins"])
        mode_ = temp[col][temp["bins"] == bins_mode].median()
        df_[col] = df_[col] / mode_
        dct_norm["all"] = mode_
        out_df = df_.copy()

    out_array = [True if row[col] >= 1 - spread and row[col] <= 1 + spread else False
                              for index, row in out_df.iterrows()]

    return out_array


def generatePairs(data):
    if not isinstance(data, list):
        try:
            data = list(data)
        except:
            raise ValueError("Input should be list or vector.")
    res = [(a, b) for idx, a in enumerate(data) for b in data[idx + 1:]]
    return res


def _normalise_data(X, method="standardscaler", copy=False):
    X = X.copy() if copy else X

    if method.lower() == "standardscaler":
        X = StandardScaler().fit_transform(X)
    elif method.lower() == "minmaxscaler":
        X = MinMaxScaler().fit_transform(X)
    elif method.lower() == "maxabsscaler":
        X = MaxAbsScaler().fit_transform(X)
    elif method.lower() == "powertransform":
        X = PowerTransformer().fit_transform(X)
    else:
        pass
    # logg.info(f"Method '{method}' not supported. Data was not normalised.")

    return X


def show_cell(data, order_by="areaNucleus", fig_height=15, fig_width=40, show_nucleus=True,
              RGB_contrasts=[3,3,4], uniqID=False, channels=None, n = None, chinfo=None, asc = True, cells = None):
    df = data.copy()

    # Ask for the number of cells to show if not provided
    if n == None:
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
    else:
        no_cells = n

    if len(df) == no_cells:
        print(f"\nShowing all cells ({len(df)}) in the selected area")

    if len(df) > no_cells:
        print('\nShowing {0} cells of a total of {1} in the selected data'.format(no_cells, str(len(df))))

    if len(df) < no_cells:
        no_cells = len(df)
        print('\nONLY ' + str(len(df)) + ' cells were found in the selected data')

    # sample cells
    new_df = df.sample(n=no_cells)


    if order_by is not None:
        new_df = new_df.sort_values(by=order_by, ascending=asc)

    # get all available channels
    all_ch = [list(chinfo[l].keys()) for l in chinfo]
    all_ch = set(sum(all_ch, ["none"]))
    if channels is None:


        # Ask for channels
        dct_colors = {'red':"", 'green':"", 'blue':""}
        for col in dct_colors:
            val = ""
            while val not in all_ch:
                val = input(f'Input channel for {col} [{"/".join(all_ch)}] ')
                print(f'Input "{val}" is not valid!') if val not in all_ch else "continue"
            dct_colors[col] = val
    else:
        dct_colors = {k: v for k, v in channels.items() if v in all_ch}
        if len(dct_colors) != len(channels):
            dropped = set(channels.values()).difference(set(dct_colors.values()))
            print(f'{len(dropped)} channel [{", ".join(dropped)}] not available and dropped')

    dct_colors = {k: v for k, v in dct_colors.items() if v != "none"}


    # Generate the figure
    figrows = ceil(no_cells/5)
    figcols = no_cells%5 if no_cells < 5 else 5

    fig, axes = plt.subplots(nrows=figrows, ncols=figcols, sharex=True, sharey=True,
                             figsize=(((int(fig_width) / 2.54)*figcols),(int(fig_height) / 2.54) * figrows))

    ax = axes.ravel()

    # Create Unique ID dictionary
    if uniqID:
        uniqID_dct = {}

    n = 0
    for index, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
        masks = np.load(join(dirname(row["path2ong"]), row["imageID"], f"{row['imageID']}_masks.npy"))
        wk_array = np.load(
            join(dirname(row["path2ong"]), row["imageID"], f"{row['imageID']}_wkarray.npy"))
        nucleus = wk_array[0].copy()
        nucleus[masks != row['cellID']] = 0
        cX_low, cX_high, cY_low, cY_high = zoomIN(nucleus, row["x_pos"], row["y_pos"], zoom_box_side=300)
        nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
        y, x = nucleus.shape
        color_red = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        color_green = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        color_blue = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        img_chan = chinfo[row['experiment']]
        subset_dct_colors = dict(filter(lambda elem: elem[1] in list(img_chan), dct_colors.items()))
        for col,ch in subset_dct_colors.items():
            channel = wk_array[img_chan[ch]].copy()
            channel = channel[cY_low:cY_high, cX_low:cX_high]
            channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            RGB = np.array((*"RGB",))
            if col == 'red':
                color = np.multiply.outer(channel, RGB == 'R')
                color_red = Image.fromarray(color).convert('L')
                enhancer = ImageEnhance.Contrast(color_red)
                color_red = enhancer.enhance(RGB_contrasts[0])
            elif col == 'green':
                color = np.multiply.outer(channel, RGB == 'G')
                color_green = Image.fromarray(color).convert('L')
                enhancer = ImageEnhance.Contrast(color_green)
                color_green = enhancer.enhance(RGB_contrasts[1])
            elif col == 'blue':
                color = np.multiply.outer(channel, RGB == 'B')
                color_blue = Image.fromarray(color).convert('L')
                enhancer = ImageEnhance.Contrast(color_blue)
                color_blue = enhancer.enhance(RGB_contrasts[2])
        mrg = Image.merge("RGB", (color_red, color_green, color_blue))
        mrg = np.array(mrg, dtype='uint8')
        scalebar = ScaleBar(0.227, 'um', box_alpha=0, location="upper left", color="w")  # 1 pixel = 1um
        if show_nucleus == False:
            kernel = np.ones((3, 3), np.uint8)
            masks[masks != int(row['cellID'])] = 0
            masks[masks == int(row['cellID'])] = 1
            masks = masks[cY_low:cY_high, cX_low:cX_high]
            masks = np.uint16(masks)
            eroded = cv2.erode(masks, kernel, iterations=2)
            nucleus = masks - eroded
            nucleus = np.array(nucleus, dtype='uint8')
            mrg[nucleus == 1] = [255, 255, 255]
            ax[n].set_xlim(0, 299)
            ax[n].set_ylim(299, 0)
            ax[n].imshow(mrg)
            ax[n].add_artist(scalebar)
        if show_nucleus == True:
            nucleus = np.ma.masked_where(nucleus == 0, nucleus)
            ax[n].set_xlim(0, 299)
            ax[n].set_ylim(299, 0)
            ax[n].imshow(mrg)
            ax[n].imshow(nucleus)
            ax[n].add_artist(scalebar)
        if uniqID:
            ax[n].set_title(f"ID: {n + 1}", fontdict={'fontsize': 8})
            uniqID_dct[str(n + 1)] = {"cellID": row["cellID"], "imageID": row["imageID"]}
        else:
            ax[n].set_title(f"{row['imageID']} | {row['cellID']}", fontdict={'fontsize': 8})
        ax[n].axis("off")
        n += 1
    plt.tight_layout()

    if uniqID:
        return fig, uniqID_dct
    else:
        return fig


def zoomIN(nucleus, x_pos, y_pos, zoom_box_side=300):
    zoom_box_side = zoom_box_side / 2
    cY = int(y_pos)
    cX = int(x_pos)
    cY_low = cY - zoom_box_side
    cY_high = cY + zoom_box_side
    cX_low = cX - zoom_box_side
    cX_high = cX + zoom_box_side
    if (cY - zoom_box_side) < 0:
        cY_low = 0
    if (cY + zoom_box_side) > len(nucleus):
        cY_high = len(nucleus)
    if (cX - zoom_box_side) < 0:
        cX_low = 0
    if (cX + zoom_box_side) > len(nucleus[0]):
        cX_high = len(nucleus[0])
    return int(cX_low), int(cX_high), int(cY_low), int(cY_high)


def embeddingPlotter(adata, basis="umap", size=20):
    df = adata.obs.copy()
    df = df.reset_index(drop=True)

    if basis == "diffmap":
        df["x"] = adata.obsm["X_diffmap"][..., 1]
        df["y"] = adata.obsm["X_diffmap"][..., 2]
    elif basis == "umap":
        df["x"] = adata.obsm["X_umap"][..., 0]
        df["y"] = adata.obsm["X_umap"][..., 1]

    f = go.FigureWidget([go.Scatter(y=df["y"],
                                    x=df["x"],
                                    mode='markers',
                                    marker=dict(size=size
                                                )
                                    )
                         ]
                        )

    scatter = f.data[0]

    t = go.FigureWidget([go.Table(
        header=dict(values=df.columns,
                    fill=dict(color='#C2D4FF'),
                    align=['left'] * 5),
        cells=dict(values=[df[col].to_list() for col in df.columns],
                   fill=dict(color='#F5F8FF'),
                   align=['left'] * 5))])

    def selection_fn(trace, points, selector):
        t.data[0].cells.values = [df.reindex(index=points.point_inds)[col] for col in df.columns]

    scatter.on_selection(selection_fn)

    return f, t


def selection2df(table):
    d = table.to_dict()
    df_out = pd.DataFrame(d['data'][0]['cells']['values'], index=d['data'][0]['header']['values']).T
    df_out = df_out.reset_index(drop=True)
    return df_out


def norm_channels(data, channel, intensity_type = "total", splitBy="experiment", method = "mode", nbins=100, showPlot=True):
    dct_norm = {}
    data = data.copy()

    col = f"{intensity_type}_intensity_{channel}"
    for exp in data[splitBy].unique():
        subset = data[data[splitBy] == exp]
        if method == "mode":
            relevant_subset = subset[[col]].copy()
            relevant_subset["bins"] = pd.cut(relevant_subset[col], nbins, duplicates="drop",
                                             labels=False)
            bins_mode = statistics.mode(relevant_subset["bins"])
            mode_ = relevant_subset[col][relevant_subset["bins"] == bins_mode].median()
            subset[col] = subset[col] / mode_
            dct_norm[exp] = mode_
        elif method == "mean":
            mean_ = subset[col].mean()
            subset[col] = subset[col] / mean_
            dct_norm[exp] = mean_
        elif method == "median":
            median_ = subset[col].median()
            subset[col] = subset[col] / median_
            dct_norm[exp] = median_

    dapi_reference = data[col].median()

    dapi_norm = {}
    for k, v in dct_norm.items():
        dapi_norm[k] = dapi_reference / v


    if showPlot:
        fig, ax = plt.subplots(figsize=(3 * len(data[splitBy].unique()), 6))
        sns.violinplot(x=splitBy, y=col, data=data, ax=ax)
        for n, exp in enumerate(data[splitBy].unique()):
            X = n
            ax.plot([X - 0.4, X + 0.4], [dct_norm[exp], dct_norm[exp]], color='r')
        plt.tight_layout()
        plt.show()

    data[col] = [row[col] * dapi_norm[row["experiment"]] for index, row in data.iterrows()]

    return data[col].to_list()

def import_ng_data(path, pattern):
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if fnmatchcase(f, pattern)]

    data_array = []
    for file in files:
        name = os.path.normpath(file).split(os.sep)[-3]
        df = pd.read_csv(file, index_col=None, header=0)
        df["experiment"] = name
        df["path2ong"] = file
        data_array.append(df)
    data = pd.concat(data_array, axis=0, ignore_index=True)
    data = remove_name_spaces(data)
    data = setcellnames(data)

    return data

def remove_name_spaces(df):
    unspaced_colnames = [name if not " " in name else name.replace(" ", "") for name in df.columns]
    if any(df.columns != unspaced_colnames):
        print("Removing spaces from variable names")
        df.colnames = unspaced_colnames
    return df


def import_channels_data(path=None, files=None):
    if files is None:
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                 f == 'channels_info.json']

    data_dict = {}
    for file in files:
        name = os.path.normpath(file).split(os.sep)[-3]
        with open(file) as json_file:
            data_dict[name] = json.load(json_file)

    return data_dict

def setcellnames(df):
    df = df.copy()

    newnames = ['-'.join(z) for z in zip(df["imageID"].astype(str), df["cellID"].astype(str))]
    df.index = newnames
    if len(set(newnames)) != len(newnames):
        # get all duplicated names
        v = df.groupby(df.index) \
            .cumcount() \
            .astype(str) \
            .str.replace('', '-', n = 1) \
            .str.replace('-0', '') \
            .values
        df.index = (df.index.values.astype(str) + v)
    return df




class Analyzor(object):

    def __init__(self, exp_dir=None, pattern="output*.csv", collated_csv=None):
        """
        Create an Analyzer object
        Parameters
        ----------
        exp_dir : string
            Path to directory containing segmented output. Will recursively import files containing
            `pattern` in its name and combines them.
        pattern : string
            Pattern of file name to import.
        collated_csv : string
            Optional- Path to collated CSV file. Useful when users have created a collated output of
            Segmentador and wish to use that instead.

        Returns
        -------
        Analyzer object

        """

        # check inputs
        if exp_dir != None:
            if not exists(exp_dir):
                raise ValueError(f"`{exp_dir}` directory does not exists")
        if collated_csv != None:
            # check file existence
            if not exists(collated_csv):
                raise ValueError(f"`{collated_csv}` file does not exists")
            # soft import to check colnames
            colnames = pd.read_csv(collated_csv, index_col=0, nrows=0).columns.tolist()
            if any(x not in colnames for x in ['path2ong', 'experiment']):
                raise ValueError(f"`{collated_csv}` was not collated properly. Please provide use `exp_dir` arg instead.")

        # import collated_csv
        if collated_csv is not None:
            dat=pd.read_csv(collated_csv)
            dat=remove_name_spaces(dat)
            dat = setcellnames(dat)
            self.data = {"raw": dat, "norm": dat.copy()}

            files = set(dat['path2ong'].to_list())
            files = [join(dirname(txt), "channels_info.json") for txt in files]
            self.meta = {}
            if all(os.path.exists(x) for x in files):
                self.meta = {"channels": import_channels_data(files=files)}
        # import individual csv
        else:
            dat = import_ng_data(exp_dir, pattern)
            self.data = {"raw": dat, "norm": dat.copy()}
            self.meta = {"channels": import_channels_data(exp_dir)}
        self.adata = ""
        self.excfeat = []
        self.buildAData()
        #self.normAData()

    def updateAData(self):
        self.buildAData(self.excfeat)
        #self.normAData()

    def excludeVars(self, vars):
        # TODO: allow pattern regex
        self.excfeat = vars
        self.excfeat = list(set(self.excfeat))
        self.updateAData()

    def includeVars(self, vars):
        # TODO: allow pattern regex
        dat_vars = self.data['norm'].copy()
        dat_vars = dat_vars.select_dtypes(include=['float64', 'int64', 'float32'])
        self.excfeat = set(dat_vars) - set(vars)
        self.excfeat = list(self.excfeat)
        self.updateAData()

    def __getitem__(self, key, data_type="norm"):
        return self.data[data_type][key].to_list()

# TODO: allow showing of data by cells
    def showData(self, vars=None, data_type='norm'):
        """
        Displays analyzer data.
        By default, this function prints out all features from the raw segmented data. To display normalized data,
        set `data_type` to 'norm'. To print out desired features, provide a list of feature names as input for `vars`.

        Parameters
        ----------
        vars : string or list of strings
            Name of features to display. Name should be found in dataframe.
        data_type : string
            Type of data to show. Can be 'raw' (default) or 'norm'.


        Returns
        -------
        None.

        """
        dat = self.data[data_type]
        if vars != None:
            if any(x not in list(dat) for x in vars):
                missingvars = list(set(vars) - set(list(dat)))
                raise ValueError(f"Variable(s) `{missingvars}` not found")
            else:
                return dat[vars]
        else:
            return dat

    def features(self):
        """
        Prints out name of features

        Returns
        -------
        None.

        """
        return self.data['raw'].columns.to_list()

    def count(self, vars, sort = False):
        """
        Counts the number of observations for a (group of) variables.

        Parameters
        ----------
        vars : string or list of strings
            Name of features to summarise. Name should be found in dataframe.

        Returns
        -------
        None.

        """
        dat = self.data['raw']
        if any(x not in list(dat) for x in vars):
            missingvars = list(set(vars) - set(list(dat)))
            raise ValueError(f"Variable(s) `{missingvars}` not found")
        return dat[vars].value_counts(sort = sort)

    def shape(self):
        """
        Prints the dimension of input data [number of cells x number of features]

        Returns
        -------
        None.

        """
        return self.data['raw'].shape

    def nfeatures(self):
        """
        Prints the number of features

        Returns
        -------
        None.

        """
        return self.data['raw'].shape[1]

    def ncells(self):
        """
        Prints the number of cells

        Returns
        -------
        None.

        """
        return self.data['raw'].shape[0]

    def normChannel(self, channel, intensity_type = "total",
                splitBy = "experiment", method = "mode", nbins = 100, showPlot = True):
        """
        Normalise DAPI/ICC channels

        Parameters
        ----------
        channel : str
            Name of channel to normalise. Value has to be found in experiment
        intensity_type : str
            Type of intensity to normalise. Can be total or avg
        splitBy : str
            Name of feature to group the cells by for normalisation (Default: experiment)
        method : str
            Method to normalise values. Can be "mode", "median" or "mean"
        nbins : int
            Number of bins to split the intensity distribution to calculate mode. Ignored if
            `Method` is not "mode"
        showPlot : bool
            Whether to display a violin plot of the pre-normalised intensity distribution

        Returns
        -------
        None.

        """

        # check if input channel is in experiment
        column = f"{intensity_type}_intensity_{channel}"
        if column in list(self.data['norm']):
            self.data['norm'][column] = norm_channels(self.data['raw'], channel, intensity_type, splitBy, method, nbins,
                                                      showPlot)
            self.updateAData()
        elif intensity_type not in ["total","avg"]:
            raise ValueError(f"`intensity_type` arguments have to be 'total' or 'avg'")
        else:
            raise ValueError(f"Channel `{channel}` not in experiment")




    def findSingleCells(self, byExperiment = True, nbins = 100, spread = 0.4, channel = None):
        """
        Annotate single cells using DAPI/ICC feature

        Parameters
        ----------
        byExperiment : bool
            Whether to annotate single cells by experiment
        nbins : int
            Number of bins to split the intensity distribution to calculate mode.
        spread : float
            Amount of spread from the mode to qualify cells as single-cell
        channel : str
            Name of nuclear marker to use. By default, will use first channel

        Returns
        -------
        None.

        """
        if channel == None:
            if len(self.meta) > 0:
                channel = list(self.meta['channels'][list(self.meta['channels'])[0]])[0]
            else:
                channel = input("Please specify nuclear marker : ")

        print(f"Using {channel} channel for qualifying single cells")
        ss_array = find_SingleCells(self.data['raw'], byExperiment, nbins, spread, channel)
        self.data['raw']['isSingleCell'] = ss_array
        self.data['norm']['isSingleCell'] = ss_array
        self.updateAData()

    def showCells(self, n=None, ch2show=None, order_by=None, ascending = True, fig_height=15, fig_width=20, show_nucleus=True,
                 RGB_contrasts=[3,3,4], uniqID=False, cells = None, filter = None):
        """
        Display image of cells

        Parameters
        ----------
        n : int
            Number of cells to display, If value is None, function will prompt for value using interactive input.
        ch2show : dict
            Dictionary of channels to display, e.g. {'red': "rfp", 'green': "beta3"}.
            If value is None, function will prompt for value using interactive input.
        order_by : string
            Feature to order cells by.
        fig_height : int
            Height of output image
        fig_width : int
            Width of output image
        show_nucleus : bool
            Whether to display nucleus
        RGB_contrasts : integerlist
            A list containing 3 integers corresponding to contrast values for Red, Green and Blue channels
        uniqID : bool
            Whether to assign unique ID for each cell
        cells : dict or str list
            Filter object by cell names. Input can be a list of cell names, or a dict output
            from `chooseCells()`
        filter : string or bool list
             Can be a string that describes the logical expression to filter cells by. E.g.
             "nuclear_area > 50" or "experiment == 'induced'". Can also be a list of boolean
             of length similar to the number of cells in object

        Returns
        -------
        None.

        """

        obj = self.copy()
        df = obj.data['norm']

        if type(cells) == dict:
            cells = cells["cells"]
            cells = np.array(cells)[[x in df.index for x in cells]]
            df = df.loc[cells,]
        elif type(cells) == list:
            cells = np.array(cells)[[x in df.index for x in cells]]
            df = df.loc[cells,]

        if type(filter) is str:
            obj.filterCells(filter = filter)

        show_cell(df, order_by, fig_height, fig_width, show_nucleus, RGB_contrasts, uniqID, ch2show, n,
                  obj.meta['channels'], ascending, cells)

    def rename(self, columns):
        """
        Rename features

        Parameters
        ----------
        columns : dict
            Dictionary datatype with old feature names as keys, and new names as values

        Returns
        -------
        None.

        """
        self.data['raw'] = self.data['raw'].rename(columns = columns, inplace = False)
        self.data['norm'] = self.data['norm'].rename(columns=columns, inplace=False)


    def plotData(self, x, y, data_type = "norm", plot_type = "scatter",
                 hue = None, alpha = 1, x_trans = None, y_trans = None,
                 x_rot = None, shuffle=False, filter = None):
        """
        Plot data from Analyzer object

        Parameters
        ----------
        x : string
            Name of feature to plot on the x-axis
        y : string
            Name of feature  to plot on the y-axis
        data_type : string
            Type of data to plot. Can be 'raw' (default) or 'norm'.
        plot_type : string
            Type of plot. Can be "scatter" (default), "violin" or "line"
        hue : string
            Name of feature to colour-code the cells by
        alpha : float
            Set opacity of scatter points. Input can take up a float value from 0 to 1.
        x_trans : string
            Scaling type to apply on x-axis. Can be "linear", "log", "symlog" or "logit"
        y_trans : bool
            Scaling type to apply on x-axis. Can be "linear", "log", "symlog" or "logit"
        x_rot : int
            Degrees to rotate x-axis labels
        shuffle : bool
            Whether to shuffle cell order. Useful to reduce overlapping of cells

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots(figsize=(8, 6))

        obj = self.copy()
        if type(filter) is str:
            obj.filterCells(filter = filter)

        dat = obj.data[data_type].copy()
        if shuffle:
            dat = dat.sample(frac=1)

        if plot_type == "scatter":
            ax = sns.scatterplot(data=dat,
                                 y=y,
                                 x=x,
                                 hue=hue,
                                 alpha=alpha)
        elif plot_type == "line":
            ax = sns.lmplot(x=x,
                       y=y,
                       data=dat,
                       hue=hue,
                       lowess=True,
                       scatter=False
                       )
        elif plot_type == "violin":
            ax = sns.violinplot(x = x,
                                 y = y,
                                 data = dat,
                                 palette = "Set3", bw = .2, hue = hue)
        if x_trans != None:
            ax.set(xscale=x_trans)
        if y_trans != None:
            ax.set(yscale=y_trans)
        if x_rot != None:
            plt.xticks(rotation=x_rot, ha="right")

        plt.tight_layout()
        plt.show()

    def plotVarDist(self, vars = "all", data_type="norm", log = False):
        """
         Plot distribution of features

         Parameters
         ----------
         vars : string or stringlist
             Name of features to plot
         data_type : string
             Whether to plot "raw" data or "norm" data

         Returns
         -------
         None.

         """


        if data_type == "scaled":
            dat = pd.DataFrame(self.adata.X.copy(), columns=self.adata.var.feature)
        else:
            dat = self.data[data_type].copy()

        if vars == "all":
            vars = list(self.adata.var.feature)
            if log:
                dat[vars] = np.log(dat[vars])
            dat.boxplot(figsize = (8,6), rot=90, column=vars)
        else:
            if log:
                dat[vars] = np.log(dat[vars])
            dat.boxplot(rot=90, column=vars, figsize = (8,6))

    def filterCells(self, filter = "", data_type = 'norm', cells = None, inplace = True):
        """
         Filter cells by feature values

         Parameters
         ----------
         filter : string or bool list
             Can be a string that describes the logical expression to filter cells by. E.g.
             "nuclear_area > 50" or "experiment == 'induced'". Can also be a list of boolean
             of length similar to the number of cells in object
         data_type : string
             Whether to plot "raw" data or "norm" data
         cells : string list
            Optional: list of string contain index of cells to retain.
         inplace : bool
            Whether to overwrite object data

         Returns
         -------
         None.

         """
        if filter != "":
            data = self.data[data_type].copy()
            if cells is not None:
                print("`filter` and `cells` arguments given, using result from `filter` only")
            if all([type(i)==bool for i in filter]) and (len(filter) == self.ncells()):
                cells = data[filter].index.to_list()
            elif type(filter) == str:
                expr_split = filter.split()
                eval_expr = "".join(["data['", expr_split[0], "']", expr_split[1], expr_split[2]])
                cells = data[eval(eval_expr)].index.to_list()
        elif type(cells) is dict:
            cells = cells['cells']

        if inplace:
            self.data['raw'] = self.data['raw'].loc[cells,]
            self.data['norm'] = self.data['norm'].loc[cells,]
            self.updateAData()
        else:
            dat = self.copy()
            dat.data['raw'] = dat.data['raw'].loc[cells,]
            dat.data['norm'] = dat.data['norm'].loc[cells,]
            dat.updateAData()

            return dat

    def copy(self):
        """
         Filter cells by feature values

         Parameters
         ----------
         expr : string or bool list
             Can be a string that describes the logical expression to filter cells by. E.g.
             "nuclear_area > 50" or "experiment == 'induced'". Can also be a list of boolean
             of length similar to the number of cells in object
         data_type : string
             Whether to plot "raw" data or "norm" data
        cells : string list
            Optional: list of string contain index of cells to retain.

         Returns
         -------
         None.

         """

        return(copy.deepcopy(self))


    def normIntensity(self, method = "mode", nbins = 100, verbose = False, hue = "experiment"):
        """
         Normalize intensity of channels

         Parameters
         ----------
         method : string
            Method to normalize the intensity by. Options are "mode" (default), "mean" or "median".
         nbins : int
             Number of bins to use.....
        verbose : boolean
            Whether to print out function messages
        hue : string
            Name of feature to colour-code the cells by

         Returns
         -------
         None.

         """
        normData, normMetadata = intensityNormalisation(self.data['norm'], method, nbins, verbose, hue)
        self.data['norm'] = normData
        self.meta['normMeta'] = normMetadata
        self.updateAData()

    def buildAData(self, excluded_features = []):
        """
         Build data for dimensional reduction

         Parameters
         ----------
         excluded_features : string or string list
            List of features to exclude from dimensional reduction

         Returns
         -------
         None.

         """

        # get numerical var from data
        dat_vars = self.data['norm'].copy()
        dat_vars = dat_vars.select_dtypes(include=['float64', 'int64', 'float32'])

        # prep vars to drop
        to_drop = ['cellID', 'x_pos', 'y_pos', 'angle','leiden',
                   'umap_1','umap_2','diffmap_1','diffmap_2','louvain']
        to_drop = [x for x in to_drop if x in list(dat_vars)]
        to_drop.extend(list(x for x in list(dat_vars) if x.endswith('_group')))
        to_drop.extend(x for x in excluded_features if x in list(dat_vars))

        dat_vars = dat_vars.drop(columns=to_drop)

        # get obs data
        dat_obs = self.data['norm'].copy()
        dat_obs = dat_obs.drop(columns=dat_vars.columns)


        # create adata
        self.adata =  anndata.AnnData(
            X = dat_vars.values,
            obs = dat_obs,
            var = pd.DataFrame(
                dat_vars.columns.to_list(),
                columns = ["feature"],
                index = [str(n) for n,c in enumerate(dat_vars.columns)])
            )
        self.adata.var_names = self.adata.var["feature"].to_list()


    def normAData(self, method = "standardscaler"):
        """
         Normalize data for dimensional reduction

         Returns
         -------
         None.

         """

        self.adata.X = _normalise_data(self.adata.X, method = method)
        sc.pp.scale(self.adata, max_value=10)

    def showAData(self):
        """
         Show AData

         Returns
         -------
         None.

         """
        return self.adata

    def showADataVars(self):
        """
         Build variables used in AData

         Returns
         -------
         None.

         """
        return self.adata.var_names.to_list()

    def showADataObs(self):
        """
         Build observations in AData


         Returns
         -------
         None.

         """
        return self.adata.obs.columns.to_list()


    def findNeighbours(self, method = "umap", n = 30, use_rep = "X"):
        """
         Finds neighbours of cells

         Parameters
         ----------
         method : string
            .....
        n : int
            text text
        use_rep : string
            text text

         Returns
         -------
         None.

         """
        sc.pp.neighbors(self.adata, n_neighbors=n, use_rep=use_rep, method=method)

    def findClusters(self, method="leiden", res = 0.6, name = None):
        """
         Cluster cells

         Parameters
         ----------
         method : string
            .....
        res : float
            text text
        name : string
            text text

         Returns
         -------
         None.

         """
        if method == "leiden":
            sc.tl.leiden(self.adata, resolution = res)
        elif method == "louvain":
            sc.tl.louvain(self.adata, resolution=res)

        if name is None:
            name = method

        self.data['raw'][name] = self.adata.obs[method].to_list()
        self.data['raw'] = self.data['raw'].astype({name: 'int'})
        self.data['norm'][name] = self.adata.obs[method].to_list()
        self.data['norm'] = self.data['norm'].astype({name: 'int'})

    def runDimReduc(self, method = "umap"):
        """
         Reduce dimension of data

         Parameters
         ----------
         method : string
            .....

         Returns
         -------
         None.

         """
        if method == "umap":
            sc.tl.umap(self.adata)
            self.data['raw']['umap_1'] = self.adata.obsm['X_umap'][...,0]
            self.data['raw']['umap_2'] = self.adata.obsm['X_umap'][..., 1]
            self.data['norm']['umap_1'] = self.adata.obsm['X_umap'][..., 0]
            self.data['norm']['umap_2'] = self.adata.obsm['X_umap'][..., 1]
        elif method == "diffmap":
            sc.tl.diffmap(self.adata)
            self.data['raw']['diffmap_1'] = self.adata.obsm['X_diffmap'][:, 1]
            self.data['raw']['diffmap_2'] = self.adata.obsm['X_diffmap'][:, 2]
            self.data['norm']['diffmap_1'] = self.adata.obsm['X_diffmap'][:, 1]
            self.data['norm']['diffmap_2'] = self.adata.obsm['X_diffmap'][:, 2]
# TODO: add color_map option
    def plotDim(self, hue = None, method = "umap", vmin = None, vmax = None):
        """
         Plots coordinates of cells on reduced dimension

         Parameters
         ----------
         method : string
            .....

         Returns
         -------
         None.

         """
        fig, ax = plt.subplots(figsize=(8, 6))

        if method == "umap":
            sc.pl.umap(self.adata, color=hue, frameon=False, ax=ax, legend_loc="on data",
                       size=30, dimensions = [0,1], vmin = vmin, vmax =vmax
                       )
        elif method == "diffmap":
            self.adata.obsm["X_diffmap_"] = self.adata.obsm["X_diffmap"][:, 1:]
            sc.pl.embedding(self.adata, "diffmap_",color=hue, frameon=False, ax=ax, legend_loc="on data",
                            size=30, dimensions=[0, 1])

        fig.tight_layout()
        plt.show()

    def runPT(self, root_cells):
        """
         Run pseudotime analysis

         Parameters
         ----------
         root : list of int
            .....


         Returns
         -------
         None.

         """
        if type(root_cells) is dict:
            root_cells = list(root_cells['index'])[0]

        self.adata.uns['iroot'] = root_cells
        sc.tl.dpt(self.adata)

    def chooseCells(self, x=None, y=None, hue=None, reduction=None, filter = None):
        obj = self.copy()

        if reduction is not None:
            x = reduction + "_1"
            y = reduction + "_2"

        if type(obj) is str:
            obj.filterCells(filter = filter)
        dat = choose_Cells(obj, x,y,hue)

        return dat

    def saveData(self, filename, format = 'csv', data_type = 'norm'):
        sep = "\t" if format == "tsv" else ","
        self.data[data_type].to_csv(filename, index=True, sep = sep)


def choose_Cells(self, x=None, y=None, hue=None):
    data = self.data['norm'].copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    if hue is not None:
        pts = ax.scatter(data[x], data[y], s=80, c=data[hue].astype('category').cat.codes)
        fc = plt.cm.jet(data[hue].astype('category').cat.codes)
    else:
        pts = ax.scatter(data[x], data[y], s=80)
        fc=None


    selector = SelectFromCollection(ax, pts, facecolors=fc)
    out = {'cells': ""}
    def accept(event):
        if event.key == "enter":
            out['index'] = selector.ind
            out['cells'] = list(data.iloc[selector.ind,].index)
            plt.close()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Select/unselect points by drawing a lasso, press `Enter` to accept")

    plt.show()

    return out




class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.1, facecolors=None):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        # if facecolors is not None: self.fc = facecolors

        line = {'color': 'grey', 'linestyle': 'dashed',
                'linewidth': 2, 'alpha': 0.8}
        self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=line)
        self.ind = []

    def update(self):
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set(alpha = self.fc[:,-1])
        self.canvas.draw_idle()


    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.xys))[0]

        if any(np.isin(ind, self.ind)):
            toremove = ind[list(np.isin(ind, self.ind))]
            self.ind = list(np.array(self.ind)[list(np.isin(self.ind,toremove, invert=True))])
            ind = list(np.array(ind)[list(np.isin(ind,toremove, invert=True))])
        self.ind.extend(np.array(ind)[list(np.isin(ind, self.ind, invert=True))])

        self.ind = list(set(self.ind))
        self.update()


    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()





