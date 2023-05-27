from typing import Tuple, Union, List
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from shapely.geometry import Polygon

from ..data.transforms.det_transforms import expand_poly

__all__ = ["DBBasePostprocess"]


class DetBasePostprocess:
    """
    Base class for all text detection postprocessings.

    Args:
        box_type (str): text region representation type after postprocessing, options: ['quad', 'poly']
        rescale_fields (list): names of fields to rescale back to the shape of the original image.
    """

    def __init__(self, box_type="quad", rescale_fields: List[str] = ["polys"]):
        assert box_type in [
            "quad",
            "poly",
        ], f"box_type must be `quad` or `poly`, but found {box_type}"
        self._rescale_fields = rescale_fields
        self.warned = False
        if self._rescale_fields is None:
            print(
                "WARNING: `rescale_filed` is None. Cannot rescale the predicted polygons to original image space"
            )

    def __call__(
        self,
        pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray],
        shape_list: Union[List, np.ndarray, ms.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Call function to execute the whole postprocessing.

        Args:
            pred (Union[Tensor, Tuple[Tensor], np.ndarray]):
                binary: text region segmentation map, with shape (N, 1, H, W)
                thresh: [if exists] threshold prediction with shape (N, 1, H, W) (optional)
                thresh_binary: [if exists] binarized with threshold, (N, 1, H, W) (optional)
            shape_list: network input image shapes, (N, 4). These 4 values represent input image [h, w, scale_h, scale_w]

        Returns:
            result (dict) with keys:
                polys: np.ndarray of shape (N, K, 4, 2) for the polygons of objective regions if region_type is 'quad'
                scores: np.ndarray of shape (N, K), score for each box
        """

        # 1. Check input type. Covert shape_list to np.ndarray
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()
        elif isinstance(shape_list, List):
            shape_list = np.array(shape_list, dtype="float32")

        if shape_list is not None:
            assert (
                len(shape_list) > 0 and len(shape_list[0]) == 4
            ), f"The length of each element in shape_list must be 4 for [raw_img_h, raw_img_w, scale_h, scale_w]. But get shape list {shape_list}"
        else:
            # shape_list = [[pred.shape[2], pred.shape[3], 1.0, 1.0] for i in range(pred.shape[0])] # H, W
            # shape_list = np.array(shape_list, dtype='float32')

            print(
                "WARNING: `shape_list` is None in postprocessing. Cannot rescale the prediction result to original image space, which can lead to inaccraute evaluatoin. You may add `shape_list` to `output_columns` list under eval section in yaml config file to address it"
            )
            self.warned = True

        # 2. Core process
        result = self.postprocess(pred, **kwargs)

        # 3. Rescale processing results
        if shape_list is not None and self._rescale_fields is not None:
            # print('Before recscale: ', result['polys'])
            # print('Shape list', shape_list)
            result = self.rescale(result, shape_list)
            # print('Postprocess rescaled the result to ', result['polys'])

        return result

    def postprocess(
        self, pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray], **kwargs
    ) -> dict:
        '''
        Core method to process network prediction without rescaling.

        Args:
            pred: same as that defined `__call__`

        Return:
            result (dict) before rescaling, with keys:
                polys: np.ndarray of shape (N, K, 4, 2) for the polygons of objective regions if region_type is 'quad'
                scores: np.ndarray of shape (N, K), score for each box
        """

        Notes:
        1. Please handle the `pred` type in implementation(since some use mindspore.nn and prefer Tensor while some use nupmpy lib and prefer numpy.
        '''
        raise NotImplementedError

    def rescale(self, result: dict, shape_list: np.ndarray) -> dict:
        """
        rescale result back to orginal image shape

        Args:
            result (dict) with keys for a data batch
                polys:  np.ndarray of shape [batch_size, num_polys, num_points, 2]. batch_size is usually 1 to avoid dynamic shape issue.

        Return:
            rescaled result specified by rescale_field
        """

        for field in self._rescale_fields:
            assert (
                field in result
            ), f"Invalid field {field}. Found fields in intermidate postprocess result are {list(result.keys())}"
            for i, sample in enumerate(result[field]):
                if len(sample) > 0:
                    result[field][i] = self._rescale_polygons(sample, shape_list[i])

        return result

    @staticmethod
    def _rescale_polygons(polygons: Union[List, np.ndarray], shape_list):
        """
        polygon: in shape [num_polygons, num_points, 2]
        shape_list: src_h, src_w, scale_h, scale_w
        """
        scale_w_h = shape_list[:1:-1]

        # print('DEBUG: rescale input: ', polygons, 'shape list: ', shape_list)
        if isinstance(polygons, np.ndarray):
            polygons = np.round(polygons / scale_w_h)
        else:
            polygons = [np.round(poly / scale_w_h) for poly in polygons]

        # print('DEBUG: rescale output: ', polygons)

        return polygons
