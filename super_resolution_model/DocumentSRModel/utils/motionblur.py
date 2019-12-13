import os
import cv2
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy import signal, misc

class BlurImage(object):
    
    def blur_image_path(self, img_path, PSFs=None, part=None, path_to_save=None, show=False):
        """
        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(img_path):
            original = misc.imread(image_path)
            result = self.blur_image(self, img, PSFs, part, path_to_save, show)
        else:
            raise Exception('Not correct path to image.')

    def blur_image(self, img, PSFs=None, part=None, path_to_save=None, show=False):
        """
        :param img: square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path_to_save: folder to save results.
        """
        img_shape = img.shape
        if len(img_shape) < 3:
            # raise Exception('We support only RGB images yet.')
            print('We support only RGB images yet.')
            return None
        elif img_shape[0] != img_shape[1]:
            # raise Exception('We support only square images yet.')
            print('We support only square images yet.')
            return None

        if PSFs is None:
            if path_to_save is None:
                PSFs = PSF(canvas=img_shape[0]).fit()
            else:
                PSFs = PSF(canvas=img_shape[0], path_to_save=os.path.join(path_to_save,
                                                                'PSFs.png')).fit(save=True)
        if part is None:
            psf = PSFs
        else:
            psf = [PSFs[part]]

        yN, xN, channel = img_shape
        key, kex = PSFs[0].shape
        delta = yN - key
        if delta < 0:
            print('resolution of image should be higher than kernel')
            return None
        # assert delta >= 0, 'resolution of image should be higher than kernel'
        result = []
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(img_shape)
                blured = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
            blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.abs(blured))

        if show:
            self.__plot_canvas(result)

        return result

    def __plot_canvas(self, result):
        if len(result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(result), figsize=(10, 10))
            if len(result) > 1:
                for i in range(len(result)):
                        axes[i].imshow(result[i])
            else:
                plt.axis('off')
                plt.imshow(result[0])

            plt.show()


class Trajectory(object):
    def __init__(self, canvas=64, iters=2000, max_len=60, expl=None, path_to_save=None):
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012]. Each
        trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration,
        is affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the
        previous particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming
        at inverting the particle velocity may arises, mimicking a sudden movement that occurs when the user presses
        the camera button or tries to compensate the camera shake. At each step, the velocity is normalized to
        guarantee that trajectories corresponding to equal exposures have the same length. Each perturbation (
        Gaussian, inertial, and impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi
        2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs
        :param canvas: size of domain where our trajectory os defined.
        :param iters: number of iterations for definition of our trajectory.
        :param max_len: maximum length of our trajectory.
        :param expl: this param helps to define probability of big shake. Recommended expl = 0.005.
        :param path_to_save: where to save if you need.
        """
        self.canvas = canvas
        self.iters = iters
        self.max_len = max_len
        if expl is None:
            self.expl = 0.1 * np.random.uniform(0, 1)
        else:
            self.expl = expl
        if path_to_save is None:
            pass
        else:
            self.path_to_save = path_to_save
        self.tot_length = None
        self.big_expl_count = None
        self.x = None

    def fit(self, show=False, save=False):
        """
        Generate motion, you can save or plot, coordinates of motion you can find in x property.
        Also you can fin properties tot_length, big_expl_count.
        :param show: default False.
        :param save: default False.
        :return: x (vector of motion).
        """
        tot_length = 0
        big_expl_count = 0
        # how to be near the previous position
        # TODO: I can change this paramether for 0.1 and make kernel at all image
        centripetal = 0.7 * np.random.uniform(0, 1)
        # probability of big shake
        prob_big_shake = 0.2 * np.random.uniform(0, 1)
        # term determining, at each sample, the random component of the new direction
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle = 360 * np.random.uniform(0, 1)

        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl

        x = np.array([complex(real=0, imag=0)] * (self.iters))

        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=ceil((self.canvas - max(x.real)) / 2), imag=ceil((self.canvas - max(x.imag)) / 2))

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x

        if show or save:
            self.__plot_canvas(show, save)
        return self

    def __plot_canvas(self, show, save):
        if self.x is None:
            raise Exception("Please run fit() method first")
        else:
            plt.close()
            plt.plot(self.x.real, self.x.imag, '-', color='blue')

            plt.xlim((0, self.canvas))
            plt.ylim((0, self.canvas))
            if show and save:
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()



class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False)
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))
        if show or save:
            self.__plot_canvas(show, save)

        return self.PSFs

    def __plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            plt.close()
            fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))
            for i in range(self.PSFnumber):
                axes[i].imshow(self.PSFs[i], cmap='gray')
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()


if __name__ == '__main__':
    folder = 'E:\\work\\proj\\relation_loc\\data\\test'
    folder_to_save = 'E:\\work\\proj\\relation_loc\\data\\test\\save'
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    blurtool = BlurImage()
    for path in os.listdir(folder):
        print(path)
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        original = misc.imread(os.path.join(folder, path))
        original = misc.imresize(original, (1024, 1024))
        blurtool.blur_image(original, PSFs=psf, part=np.random.choice([1, 2, 3]), 
                            path_to_save=folder_to_save, show=True)
        # BlurImage(os.path.join(folder, path), PSFs=psf,
        #           path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
        #     blur_image(show=True)
