import ngtools.ng_tools as ngt


def runNGS(dir, channels, dnamarker = "dapi",
           informat = ".czi", segmethod = "cellpose", useGPU = False,
           xscale = 0.454, yscale = 0.454, outdir = None, channelsinname = False):
    """
    A wrapper to perform segmentation on all images for each experiment

    Parameters
    ----------
    path : str
        Path to experiment directory. This directory should contain subdirectories
        with images for different conditions (e.g. Ki67_Cellcyle which contain 8 folders)
    channels: string array
        A 4-element array describing the channels for the entire experiment.
        (e.g. ["DAPI", "Beta3", "RFP", "GFAP"]). If different images have different
        channel assignments, set this to None and switch channelsinname to True.
    dnamarker: str
        Name of the channel that represents the DNA marker
    xscale: float
        In situations where the metadata for the x-axis scale is not present, it will
        use this given scale value. Default: 0.454
    yscale: float
        In situations where the metadata for the y-axis scale is not present, it will
        use this given scale value. Default: 0.454
    outdir: str
        Path to output directory. If not given, outputs will be saved at the same
        directory as input. If a valid path is provided, the outputs will be saved
        in the specified directory, with the same file structures as in input.
    channelsinname: Bool
        True/False as to whether is the channel assignment is in the name of
        subdirectories. This is useful for experiments with different channels
        assigned. IMPORTANT: The format has to be as such
        "ANYDESCRIPTION_Channel1,Channel2,Channel3,Channel4"

    Returns
    -------
    output.csv files in a directory called 'out_ng'
    in input directory out outdir if given. The working image and masks are
    also exported for each image, together with channel information


    """
    print(dir)
    try:
        ngs = ngt.NuclearGame_Segmentation(dir, outdir)
    except NotADirectoryError:
        print(f"Directory {dir} does not exists")
    except ValueError:
        print(f"Directory {dir} do not contain supported file formats.")
    if channelsinname:
        channels = dir.split("_")[1].split(",")
    ngs.get_file_name(_format=informat, getall=True)
    ngs.read_files()
    ngs.identify_channels(channels=channels, marker=dnamarker)
    ngs.nuclear_segmentation(method=segmethod, diameter=30, gamma_corr=True, gamma=0.25, dc_scaleCorr=1.9, GPU=useGPU)
    ngs.nuclear_features(xscale, yscale)
    ngs.add_nuclear_features()
    ngs.find_dna_peaks(box_size=10, zoom_box_size=200)
    ngs.find_dna_dots(zoom_box_size=200)
    ngs.spatial_entropy(d=5, zoom_box_size=200)
    ngs.markerGroup(n_groups=5, sample_size=10)
    ngs.saveArrays()
    ngs.saveChannelInfo()
    ngs.export_csv(filename="output.csv")

def multiNGSwrapper(path, channels, dnamarker = "dapi", xscale = 0.454, yscale = 0.454, outdir = None, channelsinname = False):
    """
    A wrapper to perform segmentation on all images for each experiment

    Parameters
    ----------
    path : str
        Path to experiment directory. This directory should contain subdirectories
        with images for different conditions (e.g. Ki67_Cellcyle which contain 8 folders)
    channels: string array
        A 4-element array describing the channels for the entire experiment.
        (e.g. ["DAPI", "Beta3", "RFP", "GFAP"]). If different images have different
        channel assignments, set this to None and switch channelsinname to True.
    dnamarker: str
        Name of the channel that represents the DNA marker
    xcale: float
        In situations where the metadata for the x-axis scale is not present, it will
        use this given scale value. Default: 0.454
    ycale: float
        In situations where the metadata for the y-axis scale is not present, it will
        use this given scale value. Default: 0.454
    outdir: str
        Path to output directory. If not given, outputs will be saved at the same
        directory as input. If a valid path is provided, the outputs will be saved
        in the specified directory, with the same file structures as in input.
    channelsinname: Bool
        True/False as to whether is the channel assignment is in the name of
        subdirectories. This is useful for experiments with different channels
        assigned. IMPORTANT: The format has to be as such
        "ANYDESCRIPTION_Channel1,Channel2,Channel3,Channel4"

    Returns
    -------
    output.csv files in a directory called 'out_ng'
    in input directory out outdir if given. The working image and masks are
    also exported for each image, together with channel information


    """
    for thisdir in os.listdir(path):
        print(thisdir)
        try:
            ngs = ngt.NuclearGame_Segmentation(join(path,thisdir), outdir)
        except NotADirectoryError:
            continue
        except ValueError:
            print(f"Directory {thisdir} do not contain supported file formats.")
            continue
        if channelsinname:
            channels = thisdir.split("_")[1].split(",")
        ngs.get_file_name(_format = informat, getall = True)
        ngs.read_files()
        ngs.identify_channels(channels = channels, marker = dnamarker)
        ngs.nuclear_segmentation(method = segmethod, diameter = 30, gamma_corr = True, gamma = 0.25, dc_scaleCorr = 1.9)
        ngs.nuclear_features(xscale, yscale)
        ngs.add_nuclear_features()
        ngs.find_dna_peaks(box_size = 10, zoom_box_size = 300)
        ngs.find_dna_dots(zoom_box_size = 300)
        ngs.spatial_entropy(d = 5, zoom_box_size = 300)
        ngs.positive2marker(frac_covered = 0.7, thresh_method = "triangle")
        ngs.assign_marker_class(n_class = 3)
        ngs.saveArrays()
        ngs.saveChannelInfo()
        ngs.export_csv(filename = "output.csv")