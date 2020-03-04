# Utility Module
Store some useful functions.
The name of the file represents the function of this file.

## Functions

##### Related to the evaluation of detection models
- `AverageMeter.py`: Computes and stores the average and current loss.
- `Detval.py`: Functions to calculate Precision and Recall.
- `Pascal_VOC.py`: Other functions to calculate Precision, Recall, and F_score.
- `polygon_wrapper.py`: Area-related operations.

##### Related to recognition models
- `average.py`: A class to calculate the average value.
- `loadData.py`: Copy the value of variable a to b.
- `strLabelConverterForAttention.py`: Encode-decode tool for recogniton tasks, especially for attention model. 
- `strLabelConverterForCTC.py`: Encode-decode tool for recogniton tasks, especially for model using CTCLoss. 

##### Other tools
- `downloadCallback.py`: As a callback function, display a progress bar during the download process.
