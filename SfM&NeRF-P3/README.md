# Building built in minutes- SfM and NeRF

Course Project-3 for RBE549 - Computer Vision (Fall 2022)

## Usage Guidelines:

### Phase 1:

1. Run
```
python3 Wrapper.py
```

#### Outputs:

1. All Intermediate Images Output are saved in Phase1-> Data-> IntermideateOutputs

#### Input and Output Data

1. You can change the savepath loacation and Data path in Arg-Parser


### Phase 2:

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


