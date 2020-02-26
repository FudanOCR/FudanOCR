from tensorboardX import SummaryWriter
import os
import torch


class Logger(object):

    def __init__(self, tag='runs/exp-1'):
        """
        Args:
            tag(string):the path to store the log file. It is based on the path to the main file.
        """
        self.writer = SummaryWriter(tag)

    def list_summary(self, tag_name, loss_list, log_freq):
        """
        Args:
            tag_name(string):Data identifier
            loss_list(list of scalar or tensor):Value to save.
            log_freq(int):Global step value to record
        """
        for index, loss in enumerate(loss_list):
            if isinstance(loss, torch.Tensor):
                self.writer.add_scalar(tag_name, loss.item(), index*log_freq + 1)
            else:
                self.writer.add_scalar(tag_name, loss, index*log_freq+1)

    def scalar_summary(self, tag_name, value, iteration_number):
        '''
        Args:
            tag_name(string):Data identifier
            value(float or tensor):Value to save. If the data is a torch scalar tensor, the function
                will extract the scalar value by x.item().
            iteration_number(int):The horizontal coordinate value of each scalar
        '''
        if isinstance(value, torch.Tensor):
            self.writer.add_scalar(tag_name, value.item(), iteration_number)
        else:
            self.writer.add_scalar(tag_name, value, iteration_number)

    def graph_summary(self, model, input_data):
        """
        Args:
            model(torch.nn.Module):Model to draw
            input_data(torch.Tensor or list of torch.Tensor):A variable or a tuple ofvariables
                to be fed
        """
        self.writer.add_graph(model, input_data)

    def image_summary(self, tag_name, img_tensor, iteration_number):
        '''
        Args:
            tag_name(string):Data identifier
            img_tensor(torch.Tensor, numpy.array, or string/blobname):An `uint8` or `float` Tensor of
                shape `[channel, height, width]` where `channel` is 1, 3(default), or 4.
                The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.You can use
                `torchvision.utils.make_grid()` to convert a batch of tensor into 3xHxW format
            iteration_number(int):The horizontal coordinate value of each scalar
        '''
        self.writer.add_image(tag_name, img_tensor, iteration_number)

    def close_summary(self):
        self.writer.close()
