import math
import numbers
import random

import cv2
import numpy as np

from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import RandomColorAdjust

__all__ = ["SVTRRecAug"]


def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude


def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude


def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation(type="random"):
    if type == "random":
        choice = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif type == "linear":
        interpolation = cv2.INTER_LINEAR
    elif type == "cubic":
        interpolation = cv2.INTER_CUBIC
    elif type == "area":
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError("Interpolation types only nearest, linear, cubic, area are supported!")
    return interpolation


class CVRandomRotation(object):
    def __init__(self, degrees=15):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        return sample_sym(degrees)

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        src_h, src_w = img.shape[:2]
        M = cv2.getRotationMatrix2D(center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

        if translate is not None:
            assert (
                isinstance(translate, (tuple, list)) and len(translate) == 2
            ), "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert (
                isinstance(scale, (tuple, list)) and len(scale) == 2
            ), "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = [shear]
            else:
                assert isinstance(shear, (tuple, list)) and (
                    len(shear) == 2
                ), "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " + "two values. Got {}".format(shear)
            )

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, height):
        angle = sample_sym(degrees)
        if translate is not None:
            max_dx = translate[0] * height
            max_dy = translate[1] * height
            translations = (np.round(sample_sym(max_dx)), np.round(sample_sym(max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = sample_uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 1:
                shear = [sample_sym(shears[0]), 0.0]
            elif len(shears) == 2:
                shear = [sample_sym(shears[0]), sample_sym(shears[1])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle, translate, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, src_h)

        M = self._get_inverse_affine_matrix((src_w / 2, src_h / 2), angle, (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        startpoints = [(0, 0), (src_w - 1, 0), (src_w - 1, src_h - 1), (0, src_h - 1)]

        def project(x, y, a, b, c):
            return int(a * x + b * y + c)

        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1])) for x, y in startpoints]

        rect = cv2.minAreaRect(np.array(endpoints))
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # add translate
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0:
            M[0, 2] += abs(translate[0])
        if translate[1] < 0:
            M[1, 2] += abs(translate[1])

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective(object):
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def get_params(self, width, height, distortion):
        offset_h = sample_asym(distortion * height / 2, size=4).astype(dtype=np.int32)
        offset_w = sample_asym(distortion * width / 2, size=4).astype(dtype=np.int32)
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])

        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return np.array(startpoints, dtype=np.float32), np.array(endpoints, dtype=np.float32)

    def __call__(self, img):
        height, width = img.shape[:2]
        startpoints, endpoints = self.get_params(width, height, self.distortion)
        M = cv2.getPerspectiveTransform(startpoints, endpoints)

        # TODO: more robust way to crop image
        rect = cv2.minAreaRect(endpoints)
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
        min_x, min_y = max(min_x, 0), max(min_y, 0)

        flags = get_interpolation()
        img = cv2.warpPerspective(img, M, (max_x, max_y), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        img = img[min_y:, min_x:]
        return img


class CVRescale(object):
    def __init__(self, factor=4, base_size=(128, 512)):
        """Define image scales using gaussian pyramid and rescale image to target scale.

        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        # assert factor is valid
        self.factor = factor
        self.base_h, self.base_w = base_size[:2]

    def __call__(self, img):
        if isinstance(self.factor, numbers.Number):
            factor = round(sample_uniform(0, self.factor))
        elif isinstance(self.factor, (tuple, list)) and len(self.factor) == 2:
            factor = round(sample_uniform(self.factor[0], self.factor[1]))
        else:
            raise RuntimeError("factor must be number or list with length 2")

        if factor == 0:
            return img
        src_h, src_w = img.shape[:2]
        cur_w, cur_h = self.base_w, self.base_h
        scale_img = cv2.resize(img, (cur_w, cur_h), interpolation=get_interpolation())
        for _ in range(factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = cv2.resize(scale_img, (src_w, src_h), interpolation=get_interpolation())
        return scale_img


class CVGaussianNoise(object):
    def __init__(self, mean=0, variance=20):
        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        if isinstance(self.variance, numbers.Number):
            variance = max(int(sample_asym(self.variance)), 1)
        elif isinstance(self.variance, (tuple, list)) and len(self.variance) == 2:
            variance = int(sample_uniform(self.variance[0], self.variance[1]))
        else:
            raise RuntimeError("degree must be number or list with length 2")

        noise = np.random.normal(self.mean, variance**0.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img


class CVMotionBlur(object):
    def __init__(self, degrees=12, angle=90):
        self.degrees = degrees
        self.angle = angle

    def __call__(self, img):
        if isinstance(self.degrees, numbers.Number):
            degree = max(int(sample_asym(self.degrees)), 1)
        elif isinstance(self.degrees, (tuple, list)) and len(self.degrees) == 2:
            degree = int(sample_uniform(self.degrees[0], self.degrees[1]))
        else:
            raise RuntimeError("degree must be number or list with length 2")
        angle = sample_uniform(-self.angle, self.angle)

        M = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class CVColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.p = p
        self.transforms = RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transforms(img)
        else:
            return img


class SVTRDeterioration(object):
    def __init__(self, variance, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if variance is not None:
            transforms.append(CVGaussianNoise(variance=variance))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            random.shuffle(self.transforms)
            transforms = Compose(self.transforms)
            return transforms(img)
        else:
            return img


class SVTRGeometry(object):
    def __init__(
        self,
        aug_type=0,
        degrees=15,
        translate=(0.3, 0.3),
        scale=(0.5, 2.0),
        shear=(45, 15),
        distortion=0.5,
        p=0.5,
    ):
        self.aug_type = aug_type
        self.p = p
        self.transforms = []
        self.transforms.append(CVRandomRotation(degrees=degrees))
        self.transforms.append(CVRandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear))
        self.transforms.append(CVRandomPerspective(distortion=distortion))

    def __call__(self, img):
        if random.random() < self.p:
            if self.aug_type:
                random.shuffle(self.transforms)
                transforms = Compose(self.transforms[: random.randint(1, 3)])
                img = transforms(img)
            else:
                img = self.transforms[random.randint(0, 2)](img)
            return img
        else:
            return img


class SVTRRecAug(object):
    def __init__(self, aug_type=0, geometry_p=0.5, deterioration_p=0.25, colorjitter_p=0.25, **kwargs):
        self.transforms = Compose(
            [
                SVTRGeometry(
                    aug_type=aug_type,
                    degrees=45,
                    translate=(0.0, 0.0),
                    scale=(0.5, 2.0),
                    shear=(45, 15),
                    distortion=0.5,
                    p=geometry_p,
                ),
                SVTRDeterioration(variance=20, degrees=6, factor=4, p=deterioration_p),
                CVColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1,
                    p=colorjitter_p,
                ),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        img = self.transforms(img)
        data["image"] = img
        return data
