from tensorboardX import SummaryWriter


class Logger(object):

    def __init__(self, log_dir):
        """
        Args:
            log_dir:the path to log file

        Create a summary writer logging to log_dir.
        """
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, loss_list, tag='train'):
        """
        Log a scalar variable.
        """
        for index, loss in enumerate(loss_list):
            self.writer.add_scalar(tag, loss, index+1)
        self.writer.close()

