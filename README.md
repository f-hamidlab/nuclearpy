# NuclearPy

## Overview of NuclearPy

NuclearPy is a suite of bioinformatics tools for 
the quantification and analysis of nuclear features from
high-content imaging data. 

## How to download NuclearPy

```bash
# Using git
git clone https://github.com/f-hamidlab/nuclearpy.git
# or if you have SSH setup
# git clone git@github.com:f-hamidlab/nuclearpy.git
```

## How to install NuclearPy
```bash
cd nuclearpy
pip install .
```

## How to run NuclearPy

### Using interactive Jupyter Notebooks

The easiest way to execute NuclearPy is by using 
our interactive Jupyter Notebooks for 
nuclei segmentation and downstream analyses. To
use NuclearPy this way, be sure to load Jupyter on
your local computer and open the Notebook you wish
to run. 

### Using command-line terminal

Alternatively, users may run NuclearPy in Terminal. 
Below, we provide a quickstart guide to performing 
nuclei segmentation on our sample image data:

```python
## load modules
import nuclearpy.segmentation as ncp
import matplotlib.pyplot as plt

## create Segmentador object
path = "../data/sample_images/experiment1"
nps = ncp.Segmentador(path, outdir = None, analyse_all=True)

## set channels and perform nuclear segmentation
nps.set_channels(channels = ["DAPI","Beta3","RFP","Ngn"], marker = "DAPI")
nps.nuclear_segmentation(method = "cellpose", 
                         diameter = 30, 
                         gamma_corr = 0.25, 
                         dc_scaleCorr = 1.9,
                         GPU = True)
                         
## quantify nuclear features
nps.nuclear_features()
nps.add_nuclear_features()
nps.find_dna_peaks(box_size = 10, zoom_box_size = 200)
nps.find_dna_dots(zoom_box_size = 200)
nps.spatial_entropy(d = 5, zoom_box_size = 200)

## output data
nps.saveArrays()
nps.saveChannelInfo()
nps.export_csv(filename = "nuclearpy_output.csv")
```
