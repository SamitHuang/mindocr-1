import os
import sys
import argparse
from typing import Union
import numpy as np
import cv2
from time import time
import mindspore as ms
import json

from predict_det import TextDetector
from predict_rec import TextRecognizer
from utils import crop_text_region, get_image_paths
from config import parse_args

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.utils.visualize import draw_bboxes, draw_bboxes_and_texts, show_imgs

#from tools.predict.text.base_predict import BasePredict
#from tools.utils.visualize import Visualization, VisMode
#from mindocr.utils.visualize import recover_image
#from tools.predict.text.utils_predict import check_args, update_config, save_pipeline_results, rescale, load_yaml


class TextSystem(object):
    def __init__(self, args):
        self.text_detect = TextDetector(args)
        self.text_recognize = TextRecognizer(args)

        self.box_type = args.det_box_type
        self.drop_score = args.drop_score
        self.save_crop_res = args.save_crop_res
        self.crop_res_save_dir = args.crop_res_save_dir
        if self.save_crop_res:
            os.makedirs(self.crop_res_save_dir, exist_ok=True)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_font_path = args.vis_font_path

    def __call__(self, img_or_path: Union[str, np.ndarray], visualize=True):
        '''
        Detect and recognize texts in an image

        Args:
            img_or_path (str or np.ndarray): path to image or image rgb values as a numpy array

        Return:
            boxes (list): detected text boxes, in shape [num_boxes, num_points, 2], where the point coordinate (x, y) follows: x - horizontal (image width direction), y - vertical (image height)
            texts (list[tuple]): list of (text, score) where text is the recognized text string for each box, and score is the confidence score.
            time_profile (dict): record the time cost for each sub-task.
        '''
        fn = os.path.basename(img_or_path).split('.')[0] if isinstance(img_or_path, str) else 'img'

        time_profile = {}
        start = time()

        # detect text regions on an image
        det_res, data = self.text_detect(img_or_path, visualize=False)

        time_profile['det'] = time() - start
        #print(det_res)
        polys = det_res['polys'].copy()
        print(f"{fn}: {len(polys)} text regions detected. Time cost: ", time_profile['det'])

        # crop text regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data['image_ori'], poly, box_type=self.box_type)
            crops.append(cropped_img)
            #print('Crop ', i, cropped_img.shape)

            if self.save_crop_res:
                cv2.imwrite(os.path.join(output_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        #show_imgs(crops, is_bgr_img=False)

        # recognize text in each crop.
        # TODO: currently run in serial to support dynamic shape input (width is dynamic) for better accuracy. Should allow batch running with fixed width for better speed.
        rs = time()
        rec_res_all_crops = self.text_recognize(crops, visualize=False)

        time_profile['rec'] = time() - rs
        print('Recognized texts: \n' +
              "\n".join([d['texts']+'\t'+str(d['confs']) for d in rec_res_all_crops]) +
              '\nTime cost: ', time_profile['rec'])

        # filter out low-score texts and merge detection and recognition results
        boxes, text_scores = [], []
        for i in range(len(polys)):
            box = det_res['polys'][i]
            box_score = det_res['scores'][i]
            text = rec_res_all_crops[i]['texts']
            text_score = rec_res_all_crops[i]['confs']
            if text_score >= self.drop_score:
                boxes.append(box)
                text_scores.append((text, text_score))

        time_profile['all'] = time() - start

        # visualize the overall result
        if visualize:
            img_shape = data['image_ori'].shape
            blank_img = np.ones([img_shape[0], img_shape[1], 3], dtype=np.uint8) * 255
            det_vis = draw_bboxes(data['image_ori'], boxes)
            text_vis = draw_bboxes_and_texts(
                                blank_img,
                                boxes,
                                [text for text, s in text_scores],
                                font_path=self.vis_font_path, #'mindocr/utils/font/simfang.ttf',
                                font_size=None,
                                hide_boxes=False,
                                text_inside_box=True)
            show_imgs([det_vis, text_vis], show=False, save_path=os.path.join(self.vis_dir, fn+'_res.png'))


        return boxes, text_scores, time_profile


def save_res(boxes_all, text_scores_all, img_paths, save_path="system_results.txt"):
    lines = []
    for i, img_path in enumerate(img_paths):
        fn = os.path.basename(img_path).split('.')[0]
        boxes = boxes_all[i]
        text_scores = text_scores_all[i]

        res = [] # result for current image
        for j in range(len(boxes)):
            res.append({"transcription": text_scores[j][0],
                    "points": np.array(boxes[j]).astype(np.int32).tolist(),
                })

        img_res_str = fn + '_' + str(i) + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
        lines.append(img_res_str)

    with open(save_path, 'w') as f:
        f.writelines(lines)
        f.close()


def main():
    # parse args
    args = parse_args()
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    ms.set_context(mode=args.mode)

    # create text system containing detector and recognizer
    text_spot = TextSystem(args)

    # warmup
    warmup = False
    if warmup:
        for i in range(2):
            text_spot(img_paths[0], visualize=False)

    # run
    tot_time = 0
    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        boxes, text_scores, time_prof = text_spot(img_path)
        tot_time += time_prof['all']

        boxes_all.append(boxes)
        text_scores_all.append(text_scores)

    fps = len(img_paths) / tot_time
    print('Average FPS: ', fps)

    # save result
    save_res(boxes_all, text_scores_all, img_paths,
             save_path=os.path.join(save_dir, 'system_results.txt'))
    print('Done! Results saved in ', save_dir)

'''
def predict_det(args):
    yaml_file = os.path.join(args.det_model_dir, 'train_config.yaml')
    det_cfg = load_yaml(yaml_file)
    det_cfg = update_config(args, det_cfg, 'det')

    det_predictor = BasePredict(det_cfg)
    vis_tool = Visualization(VisMode.crop)
    t0 = time()
    det_pred_outputs = det_predictor()

    ori_img_path_list = det_pred_outputs['img_paths']
    image_list = det_pred_outputs['pred_images']  # image_list = [image1, image2, ...]
    box_list = det_pred_outputs['preds']  # box_list = [[[(boxes1, scores1)]], [[(boxes2, scores2)]], ...]

    # cropped_images_dict = {}
    box_dict = {}
    for idx, image in enumerate(image_list):
        original_img_path = ori_img_path_list[idx].asnumpy()[0]
        # original_img_filename = os.path.splitext(os.path.basename(original_img_path))[0]
        original_img_filename = os.path.basename(original_img_path)

        image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
        image = recover_image(image)
        cropped_images = vis_tool(image, box_list[idx][0][0])
        # cropped_images_dict[original_img_filename] = cropped_images
        box_dict[original_img_filename] = {} # nested dict. {img_0:{img_0_crop_0: box array, img_0_crop_1: box array, ...}, img_1:{...}}

        # save cropped images
        if args.crop_save_dir:
            for i, crop in enumerate(cropped_images):
                crop_save_filename = original_img_filename + '_crop_' + str(i) + '.jpg'
                box_dict[original_img_filename][crop_save_filename] = box_list[idx][0][0][i]
                cv2.imwrite(os.path.join(args.crop_save_dir, crop_save_filename), crop)

    det_pred_outputs['predicted_boxes'] = box_dict
    t1 = time()
    det_time = t1 - t0
    # print(f'---det time: {det_time}s')
    # print(f'---det FPS: {len(image_list) / det_time}')
    return det_pred_outputs

def predict_old(args, det_pred_outputs):
    rec_cfg = load_yaml(args.rec_config_path)
    rec_cfg = update_config(args, rec_cfg, 'rec')

    rec_cfg.predict.loader.batch_size = 1 # TODO
    rec_predictor = BasePredict(rec_cfg)
    t2 = time()

    # rec_predict_image, rec_result, rec_img_path_list = rec_predictor()
    rec_pred_outputs = rec_predictor()
    text_list = [r['texts'][0] for r in rec_pred_outputs['preds']]

    rec_img_path_list = [os.path.basename(path.asnumpy()[0]) for path in rec_pred_outputs['img_paths']]
    rec_text_dict = dict(zip(rec_img_path_list, text_list))
    t3 = time()
    rec_time = t3 - t2
    # print(f'---rec time: {rec_time}s')
    # print(f"---rec FPS: {len(det_pred_outputs['pred_images']) / rec_time}")

    if args.result_save_dir:
        save_pipeline_results(det_pred_outputs['predicted_boxes'], rec_text_dict, args.result_save_dir)

    # if args.vis_result_save_dir:
    #     vis_tool = Visualization(VisMode.bbox_text)
    #     for idx, image in enumerate(image_list):
    #         # original_img_path = img_path_list[idx].asnumpy()[0]
    #         original_img_filename = os.path.splitext(ori_img_path_list[idx])[0]
    #         pl_vis_filename = original_img_filename + '_pl_vis' + '.jpg'
    #         image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
    #         image = recover_image(image)
    #         box_text = vis_tool(recover_image(image), box_dict[original_img_filename], text_list, font_path=args.vis_font_path) # TODO: box_dict
    #         cv2.imwrite(os.path.join(args.vis_result_save_dir, pl_vis_filename), box_text)

    return text_list

def det_rec_batch():
    args = parse_args()
    det_pred_outputs = predict_det(args)
    print('Detection finished!')
    det_pred_outputs = rescale(det_pred_outputs)
    print('Rescale finished!')
    text_list = predict_rec(args, det_pred_outputs)
    print('Detection and recognition finished!!!')
'''

if __name__ == '__main__':
    main()

