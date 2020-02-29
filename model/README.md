# Model module

This system integrates several OCR methods, including detection, identification, and end-to-end framework, designed to provide convenience to researchers. The system includes models used in the 2019 ICDAR competition, as well as models used by graduation thesis of brothers and sisters. The experimental data of the model is recorded in a shared document, which is linked as follows:
#### [Shared documents recording experimental data](https://docs.qq.com/desktop/mydoc/folder/aE338MoFVm_100001)

## Useful files
- `/documents`: Environment configuration document, use `pip` to install.
- `/detection_model`: Detection model.
- `/recognition_model`: Recognition mode.
- `/end2end_model`: End-to-end model combining detection and recognition.
- `/super_resolution_model`: Super resolution model.
- `/maskrcnn_benchmark_architecture`: Model using open source architecture
- `modelDict.py`: This function is used to get the model network. 

## Other files
Some files from the previous version of FudanOCR, which are no longer used.
`README_Chinese.md` is the README file of the previous version.
- `/config`: Config files.
- `/toolkit`: Several useful utility functions.
- `/technical_report`: Technical report, including Fudan's previous OCR technical reports.
- `/train`:  The main method import the training model from this folder.
- `/test`: The main method import the test model from this folder.
- `/demo`: Visualizing experimental results.
- `train_entry.py`: Use `python train_entry.py --config_file xxx.yaml` to train the model.
- `test_entry.py`: Use `python test_entry.py --config_file xxx.yaml` to test the model.
- `demo_entry.py`: Use `python demo_entry.py --config_file xxx.yaml` to show the demo.

## Usage
modelDict.py:
```python
def getModel(model_name):
    if model_name == 'MORAN':
        from model.recognition_model.MORAN_V2.models.moran import newMORAN
        return newMORAN

    ''' Add your network as follow '''
    elif model_name == 'YourModelName':
        from model.recognition_model.YourNetworkFile import NetworkName
        return Networkname

    else:
        return None
```