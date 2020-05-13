import torch
import cv2
import numpy as np
import types
import random
from skimage.transform import resize


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, segmaps=None):
        for t in self.transforms:
            image, segmaps = t(image, segmaps)
        return image, segmaps


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, image, segmaps):
        return self.lambd(image, segmaps)


class ConvertToFloat(object):
    def __call__(self, image, segmaps=None):
        return image.astype(np.float32), segmaps


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, segmaps=None):
        image = image.astype(np.float32)
        image /= 255
        if self.mean is not None:
            image -= self.mean
            image /= self.std
        return image.astype(np.float32), segmaps


class Resize(object):
    def __init__(self, input_shape, output_shape=None):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, image, segmaps=None):
        image = resize(image, self.input_shape, preserve_range=True)
        if self.output_shape:
            segmaps = [
                resize(
                    segmap,
                    self.output_shape,
                    preserve_range=True,
                    order=0,
                    anti_aliasing=False,
                )
                for segmap in segmaps
            ]
        return image, segmaps


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, segmaps=None):
        if random.random() < 0.5:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, segmaps


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, segmaps=None):
        if random.random() < 0.5:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, segmaps


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )

    def __call__(self, image, segmaps=None):
        if random.random() < 0.5:
            swap = self.perms[random.randint(0, len(self.perms) - 1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, segmaps


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, segmaps=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "RGB" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == "BGR" and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == "HSV" and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, segmaps


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, segmaps=None):
        if random.random() < 0.5:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, segmaps


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, segmaps=None):
        if random.random() < 0.5:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, segmaps


class ToTensor(object):
    def __call__(self, image, segmaps=None):
        return (
            np.transpose(image, (2, 0, 1)).astype(np.float32),
            [segmap.astype(int) for segmap in segmaps],
        )


class RandomMirror(object):
    def __call__(self, image, segmaps):
        if random.random() < 0.5:
            image = image[..., ::-1, :]
            for idx, segmap in enumerate(segmaps):
                segmaps[idx] = segmap[..., ::-1, :]
        return image, segmaps


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform="HSV"),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current="HSV", transform="RGB"),  # RGB
            RandomContrast(),  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, segmaps):
        im = image.copy()
        im, segmaps = self.rand_brightness(im, segmaps)
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, segmaps = distort(im, segmaps)
        return self.rand_light_noise(im, segmaps)
