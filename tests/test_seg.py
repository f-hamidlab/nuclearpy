import pytest
import filecmp
import shutil
from ngtools.wrappers import runNGS

def test_runngs():
	runNGS("data/sample_data/induced", channels=["dapi", "beta3", "rfp", "ngn"], outdir="data/test_out")
	assert filecmp.cmp("data/sample_output/out_ng/output.csv", 'data/test_out/induced/out_ng/output.csv'), "test failed"
	shutil.rmtree("data/test_out")
