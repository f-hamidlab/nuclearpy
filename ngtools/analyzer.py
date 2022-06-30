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
import pandas as pd
import os, json
from os.path import isfile, join, isdir
import statistics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scanpy as sc
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
              RGB_contrasts=[3,3,4], uniqID=False, channels=None, n = None, chinfo=None):
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

    # get all available channels
    all_ch = [list(chinfo[l].keys()) for l in chinfo]
    all_ch = set(sum(all_ch, ["None"]))
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

    dct_colors = {k: v for k, v in dct_colors.items() if v != "None"}


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
        img_chan = chinfo[row['experiment']]
        for col,ch in dct_colors.items():
            channel = wk_array[img_chan[ch] + 1].copy()
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

    data["avg_intensity_dapi"] = [row["avg_intensity_dapi"] * dapi_norm[row["experiment"]] for index, row in
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
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if f == 'output.csv']

    data_array = []
    for file in files:
        name = file.split("/")[-3]
        df = pd.read_csv(file, index_col=None, header=0)
        df["experiment"] = name
        df["path2ong"] = file
        data_array.append(df)
    data = pd.concat(data_array, axis=0, ignore_index=True)

    unspaced_colnames = [name if not " " in name else name.replace(" ","") for name in data.columns]
    if any(data.columns != unspaced_colnames):
        print("Removing spaces from variable names")
        data.colnames = unspaced_colnames

    return data

def import_channels_data(path=None, files=None):
    if files is None:
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                 f == 'channels_info.json']

    data_dict = {}
    for file in files:
        name = file.split("/")[-3]
        with open(file) as json_file:
            data_dict[name] = json.load(json_file)

    return data_dict


class NuclearGame_Analyzer(object):

    def __init__(self, exp_dir=None, csv=None, ):
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

        if csv is not None:
            dat=pd.read_csv(csv)
            self.data = {"raw": dat, "norm": ""}

            files = set(dat['path2ong'].to_list())
            files = [txt.replace("output.csv","channels_info.json") for txt in files]
            self.meta = {"channels": import_channels_data(files = files)}
        else:
            self.data = {"raw": import_ng_data(exp_dir), "norm": ""}
            self.meta = {"channels": import_channels_data(exp_dir)}
        self.adata = ""


    def showData(self, data_type = 'raw', vars = None):
        dat = self.data[data_type]
        if vars != None:
            print(dat[vars])
        else:
            print(dat)

    def colnames(self):
        print(self.data['raw'].columns)

    def count(self, vars):
        dat = self.data['raw']
        print(dat[vars].value_counts())

    def dim(self):
        print(self.data['raw'].shape)

    def ncol(self):
        print(self.data['raw'].shape[1])

    def nrow(self):
        print(self.data['raw'].shape[0])

    def ctrDAPI(self, splitBy = "experiment", nbins = 100, showPlot = True):
        self.data['raw'] = centerDAPI(self.data['raw'], splitBy, nbins, showPlot)

    def findSingleCells(self, byExperiment = True, nbins = 100, spread = 0.4, channel = None):
        if channel == None:
            channel = "dapi"
        self.data['raw'] = find_SingleCells(self.data['raw'], byExperiment, nbins, spread, channel)

    def showCell(self, n=None, ch2show=None, order_by=None, fig_height=15, fig_width=40, show_nucleus=True,
                 RGB_contrasts=[3,3,4], uniqID=False):
        show_cell(self.data['raw'], order_by, fig_height, fig_width, show_nucleus, RGB_contrasts, uniqID, ch2show, n, self.meta['channels'])

    def plotData(self, x, y, data_type = "raw", plot_type = "scatter",
                 hue = None, alpha = 1, x_trans = None, y_trans = None,
                 x_rot = None, shuffle=False):
        #fig, ax = plt.subplots(figsize=(6.4, 4.8))

        dat = self.data[data_type].copy()
        if shuffle:
            dat = dat.sample(frac=1)

        if plot_type == "scatter":
            fig = sns.scatterplot(data=dat,
                                 y=y,
                                 x=x,
                                 hue=hue,
                                 alpha=alpha)
        elif plot_type == "line":
            fig = sns.lmplot(x=x,
                       y=y,
                       data=dat,
                       hue=hue,
                       lowess=True,
                       scatter=False
                       )
        elif plot_type == "violin":
            fig = sns.violinplot(x = x,
                                 y = y,
                                 data = dat,
                                 palette = "Set3", bw = .2, hue = hue)
        if x_trans != None:
            fig.set(xscale=x_trans)
        if y_trans != None:
            fig.set(yscale=y_trans)
        if x_rot != None:
            plt.xticks(rotation=x_rot, ha="right")

        plt.tight_layout()
        plt.show()

    def plotVarDist(self, vars = "all", data_type="raw"):
        if vars == "all":
            self.data[data_type].boxplot()
        else:
            self.data[data_type].boxplot(rot=45, column=vars)

    def filterCells(self, expr = "", data_type = 'raw', cells = None):

        if expr != "":
            data = self.data[data_type].copy()
            if cells is not None:
                print("`expr` and `cells` arguments given, using result from `expr` only")
            if type(expr) == list:
                len_expr = len(expr)
                if len_expr == self.nrow():
                    cells = data[expr].index.to_list()
                else:
                    raise Exception("TEST")
            if type(expr) == str:
                expr_split = expr.split()
                eval_expr = "".join(["data['", expr_split[0], "']", expr_split[1], expr_split[2]])
                cells = data[eval(eval_expr)].index.to_list()

        self.data['raw'] = self.data['raw'].loc[cells]
        if self.data['norm'] != "":
            self.data['norm'] = self.data['norm'].loc[cells]

    def normIntensity(self, method = "mode", nbins = 100, verbose = False, hue = "experiment"):
        normData, normMetadata = intensityNormalisation(self.data['raw'], method, nbins, verbose, hue)
        self.data['norm'] = normData
        self.meta = normMetadata

    def buildAData(self, excluded_features = []):

        to_drop = ['cellID', 'x_pos', 'y_pos', 'angle']
        to_drop.extend(list(x for x in list(self.data['norm']) if x.endswith('_group')))
        to_drop.extend(excluded_features)

        # get numerical var from data
        dat_vars = self.data['norm'].copy()
        dat_vars = dat_vars.select_dtypes(include=['float64', 'int64'])
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


    def normAData(self):
        self.adata.X = _normalise_data(self.adata.X)
        sc.pp.scale(self.adata, max_value=10)

    def showAData(self):
        print(self.adata)

    def showADataVars(self):
        print(self.adata.var_names.to_list())

    def showADataObs(self):
        print(self.adata.obs.columns.to_list())


    def findNeighbours(self, method = "umap", n = 30, use_rep = "X"):
        sc.pp.neighbors(self.adata, n_neighbors=n, use_rep=use_rep, method=method)

    def findClusters(self, method="leiden", res = 0.6, name = None):
        if method == "leiden":
            sc.tl.leiden(self.adata, resolution = res)
        elif method == "louvain":
            sc.tl.louvain(self.adata, resolution=res)

        if name is None:
            name = method

        self.data['raw'][name] = self.adata.obs[method].to_list()
        self.data['norm'][name] = self.adata.obs[method].to_list()

    def runDimReduc(self, method = "umap"):
        if method == "umap":
            sc.tl.umap(self.adata)
        elif method == "diffmap":
            sc.tl.diffmap(self.adata)

    def plotDim(self, hue = None, method = "umap"):
        fig, ax = plt.subplots(figsize=(4, 4))

        if method == "umap":
            sc.pl.umap(self.adata, color=hue, frameon=False, ax=ax, legend_loc="on data",
                       size=30, dimensions = [0,1]
                       )
        elif method == "diffmap":
            sc.pl.diffmap(self.adata, color=hue, frameon=False, ax=ax, legend_loc="on data",
                       size=30, dimensions = [0,1]
                       )

        fig.tight_layout()
        plt.show()

    def runPT(self, root):
        self.adata.uns['iroot'] = root
        sc.tl.dpt(self.adata)









