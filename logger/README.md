# Logger Module

## info.py

### Write the string to the file in specified path.

Open a file for append. If the file already exists, the file pointer will be placed at the end of the file. That is, the new content will be written to the existing content. If the file or path does not exist, create a new one to write to.

P.S. file_name needs to include filename extension


	file_summary(path, file_name, content, encoding='utf8')
	'''
	Args:
		path(string):the path to the file needed to write in.
		file_name(string):the name to the file. e.g.`test.txt`
		content(string):the string needed to write.
		encoding(string):the coding scheme of the file.
	'''	


## logger.py

### Create a summary writer

Before logging anything, we need to create a writer instance. This can be done with:

	Logger = Logger(tag='runs/exp-1')
	```
	Args:
		tag(string):the path to store the log file. It is based on the path to the main file. 
	``` 

### Add a list of scalars

Input a list of scalars such as loss and write them to the log file. 

	Logger.list_summary(tag_name, loss_list, log_freq)
	```
	Args:
		tag_name(string):Data identifier
		loss_list(list of float or tensor):Value to save. If the data is a torch scalar tensor, the 
			function will extract the scalar value by x.item().
		log_freq(int):Global step value to record
	```

### Add scalar

Similar to the former 'Add a list of scalar', but the data in this function is not list but a scalar.

	Logger.scalar_summary(tag_name, value, iteration_number)
	```
	Args:
		tag_name(string):Data identifier
		value(float or tensor):Value to save. If the data is a torch scalar tensor, you need to 
			extract the scalar value by x.item()
		iteration_number:The horizontal coordinate value of each scalar
	```
	
### Add graph

Visualize a net model.

	Logger.graph_summary(model, input_data)
	```
	Args:
		model(torch.nn.Module):Model to draw
		input_data(torch.Tensor or list of torch.Tensor):A variable or a tuple ofvariables to be fed.
	```
![](https://i.imgur.com/iXv7jgG.jpg)
### Add image

Visualize a image.

Attention:this function requires the `pillow` package.

	Logger.image_summary(tag_name, img_tensor, iteration_number)
	```
	Args:
		tag_name(string):Data identifier
		img_tensor(torch.Tensor, numpy.array, or string/blobname):An `uint8` or `float` Tensor of 
			shape `[channel, height, width]` where `channel` is 1, 3(default), or 4. 
			The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8).
            Users are responsible to scale the data in the correct range/type.You can use 
			`torchvision.utils.make_grid()` to convert a batch of tensor into 3xHxW format
	```



## TensorboardX


- Build ssh tunnel in XShell, which implements the monitoring to the remote port `6006`(tensorboardX default), while the local port could be any unappropriated port such as 127.0.0.1	


- After generating log file, input `tensorboard –logdir=LOGGER_DIR` in XShell command-line interface. The `LOGGER_DIR` is the path to log file.

- Input the given URL `http://localhost:6006/` in local browser to open user interface of tensorboardX.
