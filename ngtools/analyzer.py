# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:52:32 2021

@author: gabrielemilioherreraoropeza
"""

#############################################
#                 Imports                   #
#############################################

import cv2
import anndata
import time
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir
import statistics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scanpy as sc
from math import log10
from skimage.filters import threshold_otsu, threshold_triangle
from statannotations.Annotator import Annotator
from skimage.filters import threshold_multiotsu
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
from matplotlib_scalebar.scalebar import ScaleBar
import operator

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
            if "avg_intensity" in col and not any(l in col.lower() for l in ["dapi", "gfap", "olig2"]):
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

    out_df["isSingleCell"] = [True if row[col] >= 1 - spread and row[col] <= 1 + spread else False
                              for index, row in out_df.iterrows()]

    return out_df


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
    else:
        pass
    # logg.info(f"Method '{method}' not supported. Data was not normalised.")

    return X


def show_cell(data, order_by="areaNucleus", fig_height=15, fig_width=40, show_nucleus=True,
              contrast_red=3, contrast_green=3, contrast_blue=4, uniqID=False, channels=["var", "rfp", "beta3"]):
    df = data.copy()

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

    new_df = df.sample(n=no_cells)

    # Get the names of the channels
    dct_channels = {}

    for ch in channels:
        dct_channels[ch] = input(f'Show {ch} (y/n): ')

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
        fig, axes = plt.subplots(nrows=no_cells, ncols=1, sharex=True, sharey=True,
                                 figsize=(int(fig_width) / 2.54, (int(fig_height) / 2.54) * no_cells))
    elif no_cells > 5 and no_cells <= 10:
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,
                                 figsize=(int(fig_width) / 2.54, (int(fig_height) / 2.54)))
    elif no_cells > 10:
        if int(str(no_cells)[-1]) > 5:
            fig, axes = plt.subplots(nrows=((no_cells // 5) + 1), ncols=5, sharex=True, sharey=True,
                                     figsize=(int(fig_width) / 2.54, (int(fig_height) / 2.54) * ((no_cells // 10) + 1)))
        elif int(str(no_cells)[-1]) <= 5 and int(str(no_cells)[-1]) > 0:
            fig, axes = plt.subplots(nrows=(no_cells // 5 + 1), ncols=5, sharex=True, sharey=True,
                                     figsize=(
                                     int(fig_width) / 2.54, (int(fig_height) / 2.54) * ((no_cells // 10) + 0.5)))
        elif int(str(no_cells)[-1]) == 0:
            fig, axes = plt.subplots(nrows=(no_cells // 5), ncols=5, sharex=True, sharey=True,
                                     figsize=(int(fig_width) / 2.54, (int(fig_height) / 2.54) * (no_cells // 10)))

    ax = axes.ravel()

    # Create Unique ID dictionary
    if uniqID:
        uniqID_dct = {}

    n = 0
    for index, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
        masks = np.load(join(row["path2ong"].replace("output.csv", ""), row["imageID"], f"{row['imageID']}_masks.npy"))
        wk_array = np.load(
            join(row["path2ong"].replace("output.csv", ""), row["imageID"], f"{row['imageID']}_wkarray.npy"))
        nucleus = wk_array[0].copy()
        nucleus[masks != row['cellID']] = 0
        cX_low, cX_high, cY_low, cY_high = zoomIN(nucleus, row["x_pos"], row["y_pos"], zoom_box_side=300)
        nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
        y, x = nucleus.shape
        color_red = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        color_green = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        color_blue = Image.fromarray(np.zeros((y, x, 3), dtype='uint8')).convert('L')
        for ch in dct_channels:
            if dct_channels[ch].lower() == 'y':
                channel = wk_array[channels.index(ch) + 1].copy()
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


def centerDAPI(data, splitBy="experiment", nbins=100, showPlot=True):
    modes_ = {}
    for exp in data[splitBy].unique():
        subset = data[data[splitBy] == exp]
        subset["bins"] = pd.cut(subset["total_intensity_dapi"], nbins, duplicates="drop", labels=False)
        bins_mode = statistics.mode(subset["bins"])
        mode_ = subset["total_intensity_dapi"][subset["bins"] == bins_mode].median()
        modes_[exp] = mode_

    dapi_reference = data["total_intensity_dapi"].median()

    dapi_norm = {}
    for k, v in modes_.items():
        dapi_norm[k] = dapi_reference / v

    data["cntrd_avg_intensity_dapi"] = [row["avg_intensity_dapi"] * dapi_norm[row["experiment"]] for index, row in
                                  data.iterrows()]

    if showPlot:
        fig, ax = plt.subplots(figsize=(3 * len(data[splitBy].unique()), 6))
        sns.violinplot(x=splitBy, y="total_intensity_dapi", data=data, ax=ax)
        for n, exp in enumerate(data[splitBy].unique()):
            X = n
            ax.plot([X - 0.4, X + 0.4], [modes_[exp], modes_[exp]], color='r')
        plt.tight_layout()
        plt.show()

    return data

def import_ng_data(path):
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              os.path.splitext(f)[1] == '.csv']

    data_array = []
    for file in files:
        name = file.split("/")[-3]
        df = pd.read_csv(file, index_col=None, header=0)
        df["experiment"] = name
        data_array.append(df)
    data = pd.concat(data_array, axis=0, ignore_index=True)

    return data


class NuclearGame_Analyzer(object):

    def __init__(self, exp_dir):
        """
        Start Nuclear Game.
        Parameters
        ----------
        exp_dir : string
            Is the path to the folder where all the microscope images that will be analysed
            are found.

        Returns
        -------
        None.

        """

        # create a dict with 3 slots
        self.data = {"raw": import_ng_data(exp_dir), "data": "", "adata": ""}
