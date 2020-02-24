# Config File
- `example.yaml`: Each model has its own YAML file. Change the different parameter values for each model in YAML file.

## Parameters

##### `BASE`: Model basic information.
- You can modify the parameters such as model name, number of workers in this section.
- The parameter `TYPE` indicates the model type. `R` means the recognition model. Don't forget to change it to `D` when running the detection model.

##### `DATASETS`: Dataset information.   
- The parameter `TYPE` records the data set type. The system uses different processing methods for different types of data sets.

##### `ADDRESS`: Address information.   
- Add all used addresses to this section. 
- Don't forget to delete unused address parameters. Or change their values to empty string, such as `CHECKPOINTS_DIR=''`
- `PRETRAIN_MODEL_DIR`: This parameter value ends with the name of the pre-trained model file.
 If this parameter value is a directory, the pre-trained model will be obtained from the cloud and stored in this directory.

##### `IMAGE`: Image size information.
- The parameters in this section record the size information of the input image and output image.

##### `FUNCTION`: Runtime options.
- Such as whether to use a pre-trained model.

##### `MODEL`: Model related parameters.
- Parameters required during model training / validation / test.

##### `THRESHOLD`: Threshold-related parameters

##### `xx_FREQ`: Frequency-Related Parameters
- Such as show frequency and save frequency.

## Others
Other parameters that cannot be classified are suggested to be recorded at the end of the document.