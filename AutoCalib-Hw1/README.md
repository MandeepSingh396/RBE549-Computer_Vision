# Zhang's Camera Calibration method implementation

Course Homework for RBE549 - Computer Vision (Fall 2022)

Master of Science in Robotics Engineering at [Worcester Polytechnic Institute](https://www.wpi.edu/)

## Requirements:

1. Opencv

2. Scipy

3. Numpy

4. Matplotlib

## Data
The Zhang’s paper relies on a calibration target (checkerboard in our case) to estimate camera intrinsic parameters. The calibration target used can be found in the file checkerboardPattern.pdf. This was printed on an A4 paper and the size of each square was 21.5mm. Thirteen images taken from a Google Pixel XL phone with focus locked can be accessed from 'CalibrationImgs' folder which we will use to calibrate.

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