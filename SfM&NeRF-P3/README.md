# Building built in minutes- SfM and NeRF

Phase 1: Reconstructed a 3D scene and simultaneously obtained the camera poses of a monocular camera from a set of images
with different view points using feature point correspondences (classical CV).

Phase 2: Used Neural Radiance fields (NeRF) to synthesize novel views of complex scenes by optimizing a continuous
volumetric scene function using a sparse set of input views (Deep Learning)



## Phase 1 - Structure from Motion:

### Input:
The data given to us is a set of 5 images of Unity Hall at WPI, using a Samsung S22 Ultra’s primary camera at f/1.8 aperture, ISO 50 and 1/500 sec shutter speed.

<img src="Phase1/Data/IntermediateOutputImages/Imgs.png"  align="center" alt="Undistorted" width="800"/>

The data folder contains 4 matching files named matching*.txt where * refers to numbers from 1 to 5. For eg., matching3.txt contains the matching between the third image and images that come after, i.e., I3↔I4,I3↔I5. This is the reason image 5 does not have a text file.

### Initial Feature matching
<img src="Phase1/Data/IntermediateOutputImages/feature_matching.png"  align="center" alt="Undistorted" width="500"/>

### Outlier Rejectiong - RANSAC (using Fundamental matrix)
<img src="Phase1/Data/IntermediateOutputImages/matching_Inliers.jpeg"  align="center" alt="Undistorted" width="500"/>

### Cheirality check
<img src="Phase1/Data/IntermediateOutputImages/Initial triangulation plot with disambiguity, showing all four possible camera poses.jpeg"  align="center" alt="Undistorted" width="500"/>

### Triangulation(1 & 2) using Correct Camera pose
<img src="Phase1/Data/IntermediateOutputImages/Projection_of_correct_camera_pose_for 2_cams.jpeg"  align="center" alt="Undistorted" width="500"/>

### After Perspective-n-Point and Bundle Adjustment
<img src="Phase1/Data/IntermediateOutputImages/bundel_adj.png"  align="center" alt="Undistorted" width="500"/>


### Usage Guidelines:

1. Run
```
python3 Wrapper.py
```

#### Outputs:

1. All Intermediate Images Output are saved in Phase1-> Data-> IntermideateOutputs

#### Input and Output Data

1. You can change the savepath loacation and Data path in Arg-Parser


## Phase 2 - NeRF:
Implementing the original NERF method [from this paper](https://arxiv.org/abs/2003.08934).

### Input:
Download the lego data for NeRF from the original author’s link [here](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)

#### Sample input

<img src="Phase2/Results/input.png"  align="center" alt="Undistorted" width="300"/>

### Neural Network used
<img src="Phase2/Results/Network.png"  align="center" alt="Undistorted" width="300"/>

### Training
<img src="Phase2/Results/loss.png"  align="center" alt="Undistorted" width="400"/>

### Result on Test set
<img src="Phase2/Results/ezgif.com-gif-maker (4).gif"  align="center" alt="Undistorted" width="500"/>

### Usage Guidelines:

#### Training:
1. Change the directory to Phase 2.
2. To train the NeRF model on GPU:

```
python3 NeRF_train.py
```
3. Output of Loss plot will be saved in Results folder.

#### Testing
1. Change the directory to Phase 2.
2. To test the model:

```
python3 NeRF_test.py
```
3. Output video will be saved in the same directory.


## References:
1. https://rbe549.github.io/fall2022/proj/p3/
2. https://arxiv.org/abs/2003.08934



