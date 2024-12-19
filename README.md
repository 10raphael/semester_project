# Semester Project Summary
Summary of the tools used during my semester project (BEHAVE, SAM, XMem & BundleSDF)

**BEHAVE**

After dowloading the sequence of your choice from the [official website](https://virtualhumans.mpi-inf.mpg.de/behave/license.html), follow the steps in [adjoint repo](https://github.com/xiexh20/behave-dataset#generate-images-from-raw-videos) of the dataset in order to generate the raw rgb and depth images. Normalize the depth images with 1269.0/depth_img.max(), rescale rgb and depth maps as well as masks to (640, 480). In the end, your BEHAVE dataset should have the same characteristics as the example milk data provided show in the [BundleSDF example](https://github.com/NVlabs/BundleSDF#run-on-your-custom-data)

The contents of the cam_K.txt file can be found on the [BEHAVE website](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip) as well. Select the desired view from `.../calibs/intrinsics/.../calibration.json` and make it of the form ([fx, 0, cx], [0, fy, cy], 0, 0, 1]]). Also rescale the calibration matrix by (640, 480).

Select between Date03\_Sub05\_boxlarge.1, Date03\_Sub05\_yogamat.3 and Date03\_Sub04\_tablesquare\_lift.3

**SAM**

Run `pip install segment-anything`, then SAM can be used via `from segment_anything import sam_model_registry, SamPredictor`, no further initialization is required. 

**XMem**

For XMem, follow the intructions in the [associated repository](https://github.com/hkchengrex/XMem/blob/main/docs/INFERENCE.md#inference). The input formatting is described below. 

**BundleSDF**

Follow these steps for [initial setup](https://github.com/NVlabs/BundleSDF#dockerenvironment-setup)

To be able to run BEHAVE sequences in BundleSDF running on a NVIDIA GeForce GTX 1080, we did the following:
1. loftr_wrapper.py:
-  line 43: reduce `batch_size = 64` to `batch_size = 8`
-  line 63: comment out total_n_matches
-  line 79: add `del tmp`

2. bundlesdf.py:
-  line 179: add `torch.cuda.empty_cache()`
-  line 456: comment out `pdb.set_trace()` (no GUI used, so not needed, led to failures)
-  line 794: add `torch.cuda.empty_cache()`
  
3. run_custom.py:
-  line 24: (comment out line 23) add `cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_behave.yml",'r'))`
  
4. nerf_runner.py:
-  line 242f: add `del embed_fn`
-  line 1492-1498: comment out, redundant
-  line 1508-1509: `if valid.sum()==0: continue` (this was pushed on the 20.11.23)

  Note that all of this was done with the BundleSDF commit of Sept. 21 2023. The modified scripts are also provided, in theory, replacing the pulled files with these one's should work. 

After successful initialization, follow the instuction to [run on you custom data](https://github.com/NVlabs/BundleSDF#run-on-your-custom-data). I only ever ran the joint tracking and reconstruction, step 1.

**My method**

First `conda create --name ... python 3.9`, then `pip -r install requirements.txt`, then finally, after modifing the path to the dataset, `python my_method.py`. Download the [model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and save it in the same directory. The results are dumped in a folder called `\output`. Take the largest mask, as written down in the  `my_method.py` log, and move is to the `\Annotations` directory, one of the 2 inputs for XMem. In a separate environment, follow these [instructions](https://github.com/hkchengrex/XMem/blob/main/docs/INFERENCE.md#on-custom-data) for using XMem. Note that the RGB sequence must be converted to .jpg. 

XMem produces a folder with object instance masks, which can be used instead of the masks from the BEHAVE dataset, as input for BundleSDF.

