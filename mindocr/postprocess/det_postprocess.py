from typing import Tuple, Union, List
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from shapely.geometry import Polygon

from ..data.transforms.det_transforms import expand_poly

__all__ = ['DBPostprocess', 'PSEPostprocess']


class DetBasePostprocess:
    """
    Base class for all text detection postprocessings.

    Args:
        box_type (str): text region representation type after postprocessing, options: ['quad', 'poly']  
        rescale_fields (list): names of fields to rescale back to the shape of the original image.
    """
    def __init__(self, 
                 box_type='quad',
                 rescale_fields: List[str] = ['polys']):

        assert box_type in ['quad', 'poly'], f'box_type must be `quad` or `poly`, but found {box_type}'
        self._rescale_fields = rescale_fields
        self.warned = False 
        if self._rescale_fields is None:
            print('WARNING: `rescale_filed` is None. Cannot rescale the predicted polygons to original image space') 

    @staticmethod
    def _rescale_polygons(polygons: Union[List, np.ndarray], shape_list):
        '''
        polygon: in shape [num_polygons, num_points, 2]
        shape_list: src_h, src_w, scale_h, scale_w 
        '''
        #src_h, src_w, scale_h, scale_w = shape_list
        scale_w_h = shape_list[:1:-1]

        print('D-- rescale input: ', polygons)
        print('shape list: ', shape_list)
        if isinstance(polygons, np.ndarray):
            #polygons[:, :, 0] = np.round(polygons[:, :, 0] / scale_w) 
            #polygons[:, :, 1] = np.round(polygons[:, :, 1] / scale_h) 
            polygons = np.round(polygons / scale_w_h)
        else:
            polygons = [np.round(poly / scale_w_h) for poly in polygons]

        print('D-- rescale output: ', polygons)

        return polygons

    def rescale(self, result: dict, shape_list: np.ndarray) -> dict:
        '''
        Args:
            result (dict) with keys for a data batch
                polys:  np.ndarray of shape [batch_size, num_polys, num_points, 2]. batch_size is usually 1 to avoid dynamic shape issue.
        '''

        for field in self._rescale_fields:
            assert field in result, f'Invalid field {field}. Found fields in intermidate postprocess result are {list(result.keys())}' 
            for i, sample in enumerate(result[field]): 
                if len(sample) > 0:
                    result[field][i] = self._rescale_polygons(sample, shape_list[i]) 

        return result

    def process(self, 
            pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray], 
            shape_list: np.ndarray = None, 
            **kwargs) -> dict:
        '''
        Core method to process network prediction. 
        '''
        raise NotImplementedError

    def __call__(self, 
            pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray], 
            shape_list: Union[List, np.ndarray, ms.Tensor]=None, 
            **kwargs) -> dict:
        """
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
        
        # 1. check input type. Covert shape_list to np.ndarray
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()
        elif isinstance(shape_list, List):
            shape_list = np.array(shape_list, dtype='float32')
        
        if shape_list is not None:
            assert len(shape_list) > 0 and len(shape_list[0])==4, f'The length of each element in shape_list must be 4 for [raw_img_h, raw_img_w, scale_h, scale_w]. But get shape list {shape_list}'
        else:
            #shape_list = [[pred.shape[2], pred.shape[3], 1.0, 1.0] for i in range(pred.shape[0])] # H, W
            #shape_list = np.array(shape_list, dtype='float32')

            print('WARNING: `shape_list` is None in postprocessing. Cannot rescale the prediction result to original image space, which can lead to inaccraute evaluatoin. You may add `shape_list` to `output_columns` list under eval section in yaml config file to address it')
            self.warned = True

        # 2. core process
        result = self.process(pred, shape_list, **kwargs)    

        # 3. rescale processing results if shape_list is given 
        if shape_list is not None and self._rescale_fields is not None:
            #print('Before recscale: ', result['polys'])
            #print('Shape list', shape_list)
            result = self.rescale(result, shape_list)
            #print('Postprocess rescaled the result to ', result['polys'])

        return result


class DBPostprocess(DetBasePostprocess):
    """
    DBNet & DBNet++ postprocessing pipeline: extracts polygons / rectangles from a binary map (heatmap) and returns
        their coordinates.
    Args:
        binary_thresh: binarization threshold applied to the heatmap output of DBNet.
        box_thresh: polygon confidence threshold. Polygons with scores lower than this threshold are filtered out.
        max_candidates: maximum number of proposed polygons.
        expand_ratio: controls by how much polygons need to be expanded to recover the original text shape
            (DBNet predicts shrunken text masks).
        output_polygon: output polygons or rectangles as the network's predictions.
        pred_name: heatmap's name used for polygons extraction.
        rescale_fields: name of fields to scale back to the shape of the original image.
    """
    def __init__(self, binary_thresh: float = 0.3, box_thresh: float = 0.7, max_candidates: int = 1000,
                 expand_ratio: float = 1.5,  box_type='quad', pred_name: str = 'binary', rescale_fields: List[str] = ['polys']
                 ):
        super().__init__(box_type, rescale_fields)

        self._min_size = 3
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._expand_ratio = expand_ratio
        self._out_poly = (box_type == 'poly') 
        self._name = pred_name
        self._names = {'binary': 0, 'thresh': 1, 'thresh_binary': 2}

    def process(self, 
                pred: Union[Tensor, Tuple[Tensor], np.ndarray], 
                shape_list: Union[Tensor, np.ndarray] = None,
                **kwargs) -> dict:
        """
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
        if isinstance(pred, tuple):
            pred = pred[self._names[self._name]]
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        pred = pred.squeeze(1)

        segmentation = pred >= self._binary_thresh

        dest_size = np.array(pred.shape[:0:-1]) - 1

        polys, scores = [], []
        for pr, segm, size in zip(pred, segmentation, dest_size):
            sample_polys, sample_scores = self._extract_preds(pr, segm, size)
            polys.append(sample_polys)
            scores.append(sample_scores)

        output = {'polys': polys, 'scores': scores}
            
        return output

    def _extract_preds(self, pred: np.ndarray, bitmap: np.ndarray, dest_size: np.ndarray):
        outs = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:  # FIXME: update to OpenCV 4.x and delete this
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        polys, scores = [], []
        for contour in contours[:self._max_candidates]:
            contour = contour.squeeze(1)
            score = self._calc_score(pred, bitmap, contour)
            if score < self._box_thresh:
                continue

            if self._out_poly:
                epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                points = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze(1)
                if points.shape[0] < 4:
                    continue
            else:
                points, min_side = self._fit_box(contour)
                if min_side < self._min_size:
                    continue

            poly = Polygon(points)
            poly = np.array(expand_poly(points, distance=poly.area * self._expand_ratio / poly.length))
            if self._out_poly and len(poly) > 1:
                continue
            poly = poly.reshape(-1, 2)

            _box, min_side = self._fit_box(poly)
            if min_side < self._min_size + 2:
                continue
            if not self._out_poly:
                poly = _box

            # TODO: an alternative solution to avoid calling self._fit_box twice:
            # box = Polygon(points)
            # box = np.array(expand_poly(points, distance=box.area * self._expand_ratio / box.length, joint_type=pyclipper.JT_MITER))
            # assert box.shape[0] == 4, print(f'box shape is {box.shape}')

            polys.append(np.clip(np.round(poly), 0, dest_size).astype(np.int16))
            scores.append(score)

        if self._out_poly:
            return polys, scores
        return np.array(polys), np.array(scores).astype(np.float32)

    @staticmethod
    def _fit_box(contour):
        """
        Finds a minimum rotated rectangle enclosing the contour.
        """
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    @staticmethod
    def _calc_score(pred, mask, contour):
        """
        calculates score (mean value) of a prediction inside a given contour.
        """
        min_vals = np.clip(np.floor(np.min(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        max_vals = np.clip(np.ceil(np.max(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        return cv2.mean(pred[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1],
                        mask[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1].astype(np.uint8))[0]


class PSEPostprocess:
    def __init__(self, binary_thresh=0.5, box_thresh=0.85, min_area=16,
                 box_type='quad', scale=4, rescale_fields=None):
        from .pse import pse
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._min_area = min_area
        self._box_type = box_type
        self._scale = scale
        self._interpolate = nn.ResizeBilinear()
        self._sigmoid = nn.Sigmoid()
        if rescale_fields is None:
            rescale_fields = []
        self._rescale_fields = rescale_fields
        self._pse = pse

    def __call__(self, pred, shape_list=None, **kwargs):  # pred: N 7 H W
        '''
        Args:
            pred (Tensor): network prediction with shape [BS, C, H, W]
            shape_list (Union[List[List[float]], np.ndarray, ms.Tensor]: a list of shape info [raw_img_h, raw_img_w, ratio_h, ratio_w] for each sample in batch  
        '''
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)
        
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()
        #if isinstance(shape_list, np.ndarray):
        #    shape_list = shape_list.tolist()

        if shape_list is not None:
            assert len(shape_list) > 0 and len(shape_list[0])==4, f'The length of each element in shape_list must be 4 for [raw_img_h, raw_img_w, scale_h, scale_w]. But get shape list {shape_list}'
        else:
            shape_list = [[pred.shape[2], pred.shape[3], 1.0, 1.0] for i in range(pred.shape[0])] # H, W
            
        pred = self._interpolate(pred, scale_factor=4 // self._scale)
        score = self._sigmoid(pred[:, 0, :, :])

        kernels = (pred > self._binary_thresh).astype(ms.float32)
        text_mask = kernels[:, :1, :, :]
        text_mask = text_mask.astype(ms.int8)

        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        score = score.asnumpy()
        kernels = kernels.asnumpy().astype(np.uint8)
        poly_list, score_list = [], []
        for batch_idx in range(pred.shape[0]):
            boxes, scores = self._boxes_from_bitmap(score[batch_idx],
                                                    kernels[batch_idx],
                                                    shape_list[batch_idx])
            poly_list.append(boxes)
            score_list.append(scores)

        return {'polys': poly_list, 'scores': score_list}

    def _boxes_from_bitmap(self, score, kernels, shape):
        label = self._pse(kernels, self._min_area)
        return self._generate_box(score, label, shape)

    def _generate_box(self, score, label, shape):
        src_h, src_w, ratio_h, ratio_w = shape
        label_num = np.max(label) + 1
        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < self._min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self._box_thresh:
                label[ind] = 0
                continue

            if self._box_type == 'quad':
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
            else:
                raise NotImplementedError(
                    f"The value of param 'box_type' can only be 'quad', but got '{self._box_type}'.")

            if 'polys' in self._rescale_fields:
                bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
                bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            boxes.append(bbox)
            scores.append(score_i)

        return boxes, scores
