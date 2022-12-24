# Zhang's Camera Calibration method implementation
Implementation of Zhang's camera calibration method from scratch to estimate the camera intrinsics and distortion parameters.


## Data
The Zhang’s paper relies on a calibration target (checkerboard in our case) to estimate camera intrinsic parameters. The calibration target used can be found in the file checkerboardPattern.pdf. This was printed on an A4 paper and the size of each square was 21.5mm. Thirteen images taken from a Google Pixel XL phone with focus locked can be accessed from 'CalibrationImgs' folder which we will use to calibrate.

## Results
                   
```
Intrinsic matrix :                   Distortion Coefficients:

[[2464.4 0.3680 0763.8],              [0.0125  -0.0125]
 [  0    2441.1 1348.3],
 [  0      0       1  ]]
```

### Reprojected corners

<img src="Results/Reprojected_corners/reprojected_corners1.png"  align="center" alt="Undistorted" length = "200" width="300"/>


## Usage Guidelines:

Open the 'AutoCalib' directory on terminal and enter the commmand:

```
python3 Wrapper.py
```

Corners on original images and Reprojected images are stored in Results folder.

Please refer the report for details of the method followed and corresponding results.


## References

1. https://rbe549.github.io/fall2022/hw/hw1/
2. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf