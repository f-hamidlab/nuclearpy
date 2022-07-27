#############################################
#     Imports 							    #
#############################################

import ngtools.segmentation as ngt
import numpy as np
import pytest


# TODO: Update test outputs

def custom_assert(actual, expected, test_name=""):
    assert actual == expected, f"\tError in {test_name}\n\texpected {expected} \n\tbut got {actual}\n\n"


#############################################
#     Segmentador tests					    #
#############################################
ngs = []
file = "Snap-120"


def test_init():
    global ngs

    # test error catching
    with pytest.raises(OSError):
        ngs = ngt.Segmentador("data/sample_images/experiment7", outdir="data/sample_output", analyse_all=True)
        ngs = ngt.Segmentador("data/sample_images", outdir="data/sample_output", analyse_all=True)

    ngs = ngt.Segmentador("data/sample_images/experiment2", outdir="data/sample_output", analyse_all=True)
    custom_assert(ngs.path_save, 'data/sample_output/experiment2/out_ng', "Set output dir")
    custom_assert(ngs.image_format, '.czi', "Setting image format")
    custom_assert(len(ngs.data["files"]), 1, "Get number of images")
    custom_assert(ngs.data["files"]["Snap-120"]["path"], 'data/sample_images/experiment2/Snap-120.czi',
                  "Get path to image file")
    # TODO: test important variables

    custom_assert(len(ngs.data["files"][file]), 3, "Get new number of keys")


# def test_read_files():
#     file = "Snap-120"
#     ngs.read_files()
#
#     # TODO: test important variables
#
#     custom_assert(len(ngs.data["files"][file]), 4, "Get new number of keys")

def test_init_other_files():
    tiff = ngt.Segmentador("data/sample_images/tiff", outdir="data/sample_output", analyse_all=True)


def test_set_channels():
    with pytest.raises(ValueError):
        ngs.set_channels(channels=["dapi", "ch1", "ch2"], marker="dapi")
        ngs.set_channels(channels=["dapi", "ch1", "ch2", "ch3"], marker="dapis")

    ngs.set_channels(channels=["dapi", "ch1", "ch2", "ch3"], marker="dapi")

    custom_assert(ngs.data["dna_marker"], "dapi", "Set DNA marker")
    custom_assert(len(ngs.data["channels_info"]), 4, "Set number of channels")

    # test interactivity
    # ngs.set_channels()
    # custom_assert(ngs.data["dna_marker"], "dapi", "Set DNA marker")
    # custom_assert(len(ngs.data["channels_info"]), 4, "Set number of channels")


def test_nuclear_seg():
    file = "Snap-120"
    ngs.nuclear_segmentation()

    custom_assert(len(ngs.data["files"][file]), 7, "Get new number of keys after segmentation")
    custom_assert(list(ngs.data["files"][file])[3:], ['working_array', 'masks', 'flows', 'th_array'],
                  "Get name of new keys")
    custom_assert(ngs.data["files"][file]['working_array'].shape, (4, 1462, 1936), "Get dimension of wk_array")
    custom_assert(ngs.data["files"][file]['masks'].shape, (1462, 1936), "Get dimension of masks")
    custom_assert(ngs.data["files"][file]['th_array'].shape, (1462, 1936), "Get dimension of th_array")

    custom_assert(len(np.unique(ngs.data["files"][file]['masks'])), 160, "Get number of seg nuclei")
    custom_assert(ngs.data["files"][file]['masks'].max(), 175, "Get highest mask index")
    custom_assert(1 in np.unique(ngs.data["files"][file]['masks']), False, "Test if masks have been filtered")


def test_nuclear_feat():
    file = "Snap-120"
    ngs.nuclear_features()

    custom_assert(len(ngs.data["files"][file]["nuclear_features"]), 14, "Get number of nuclear features")

    first_cell_info = [ngs.data["files"][file]["nuclear_features"][ft][0] for ft in
                       ngs.data["files"][file]["nuclear_features"]]
    expected_info = [6, 1802, 112, 41, 14.3, 10.1, 0.707, 0.853, 0.708, 0.966, 164, 34, -1.431, 'Snap-120']
    custom_assert(first_cell_info, expected_info, "Get info for first cell")


def test_add_nuc_feat():
    ngs.add_nuclear_features()

    file = "Snap-120"
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]), 31, "Get new number of nuclear features")


def test_dna_peaks_dots():
    ngs.find_dna_peaks()
    ngs.find_dna_dots()

    file = "Snap-120"
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]), 34, "Get new number of nuclear features")
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]["dna_peaks"]),
                  len(ngs.data["files"][file]["nuclear_features"]["cellID"]),
                  "Test length of dna_peaks")
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]["dna_dots"]),
                  len(ngs.data["files"][file]["nuclear_features"]["cellID"]),
                  "Test length of dna_dots")
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]["dna_dots_size_median"]),
                  len(ngs.data["files"][file]["nuclear_features"]["cellID"]),
                  "Test length of dna_dots_size_median")
    first_cell_dots_info = [ngs.data["files"][file]["nuclear_features"][ft][0] for ft in
                            ["dna_peaks", "dna_dots", "dna_dots_size_median"]]
    custom_assert(first_cell_dots_info, [11, 9, 1.7], "Test values for first cell")


def test_spat_entropy():
    ngs.spatial_entropy()
    file = "Snap-120"

    custom_assert(len(ngs.data["files"][file]["nuclear_features"]["spatial_entropy"]),
                  len(ngs.data["files"][file]["nuclear_features"]["cellID"]),
                  "Test length of spatial_entropy")
    custom_assert(list(ngs.data["files"][file]["nuclear_features"]["spatial_entropy"][0:3]), [3.761, 3.888, 3.766],
                  "Test spat_entropy values for first 3 cells")


def test_marker_group():
    ngs.markerGroup(n_groups=5)
    file = "Snap-120"
    custom_assert(len(ngs.data["files"][file]["nuclear_features"]), 38, "Get new number of nuclear features")

    first_5_cell_group_info = [ngs.data["files"][file]["nuclear_features"][ft][0:5] for ft in
                               ["ch1_group", "ch2_group", "ch3_group"]]
    custom_assert(first_5_cell_group_info, [[2, 1, 1, 1, 1], [2, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                  "Get info for first 5 cells")


def test_final_obj_length():
    lst_fts = ngs.get_lst_features()
    custom_assert(len(lst_fts), 38, "Get final number of nuc features")

    dct_df = {}

    for ft in lst_fts:
        dct_df[ft] = [l for file in ngs.data["files"] for l in ngs.data["files"][file]["nuclear_features"][ft]]

    lengths = [len(dct_df[ft]) for ft in lst_fts]
    custom_assert(lengths == [159] * 38, True, "Test lengths of outputs")
