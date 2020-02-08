# Gender Classification

## Description
Using the LibriSpeech dataset to train a model that could classify genders(male or female)

## Environment
python: virtual environment with version 3.6
```buildoutcfg
# activate the virtual environment(under the project folder)
source venv/bin/activate
```
pycharm IDE

Libraries:
pytorch: neural model
librosa: MFcc feature extraction
 
## Files and folders
1. constants.py: definition of project and file paths, num of classes, num of MFCCs, num of frames, etc.
2. preprocessings.py: methods of loading data and performing MFCC(using the library: librosa)
3. GenderClassifier.py: the LSTM model, including 2 conv layers - 1 lstm layer - 3 full connected layers
4. main.py: main file in which data is loaded and the model is instantiated, trained and tested
5. postprocessings.py: extracting weights from the model, these weights need to be quantized
```python
# run the model using
python main.py
```

1. LibriSpeech/: dataset, using the dev-clean dataset(337M before extraction)
2. pkl_data/: storing the features extracted after MFCC
3. weights_data/: extracted weights from the model

## Notes
Some parameters in the main.py file, 

feature_loading: determine to load features from the original data(\*.flac files, much slower) or from the pkl files(\*.pkl files, faster)

feature_store: determine whether to store the features in pkl files after loading from the original data

actions: 
1. First time loading the original data, doing the MFCC, then storing into pkl files
2. Second time and after, directly loading features from the pkl files
```python
# first time
feature_loading = load_from_flac
feature_store = True

# second time and after
feature_loading = load_from_pkl
```


## Todos
1. Parameter quantization - from floating point to fixed point
2. Compress the model using circulant matrix
3. 


## Project log

2-3-2020
---
Basic model completed, accuracy = 94%

2-5-2020
---
Modications on the model, num of parameters get reduced, accuracy = 93%

Model's structure:
```
GenderClassifier(
  (conv): Sequential(
    (0): Conv1d(40, 16, kernel_size=(4,), stride=(1,)), 2,576 params
    (1): ReLU(), 0 params
    (2): Conv1d(16, 8, kernel_size=(2,), stride=(1,)), 264 params
    (3): ReLU(), 0 params
    (4): MaxPool1d(kernel_size=16, stride=16, padding=0, dilation=1, ceil_mode=False), 0 params
  ), 2,840 params
  (lstm): LSTM(8, 16, batch_first=True), 1,664 params
  (fc): Sequential(
    (0): Flatten(), 0 params
    (1): Linear(in_features=80, out_features=64, bias=True), 5,184 params
    (2): Linear(in_features=64, out_features=32, bias=True), 2,080 params
    (3): Linear(in_features=32, out_features=2, bias=True), 66 params
    (4): Softmax(dim=1), 0 params
  ), 7,330 params
), 11,834 params
```

2-7-2020
---
1. quantized parameters are checked(fixed point weight and bias data), accuracy = 91%
2. approximate sigmoid and tanh functions are checked, accuracy drops obviously, accuracy -> 75%~80%