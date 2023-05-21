# MindOCR Online Inference

## Text Detection

To run text detection on an input image or multiple images in a directory, please run:

```shell
python tools/infer/text/predict_det.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm DB++
```

For more argument illustrations and usage, please run `python tools/infer/text/predict_det.py -h` or view `tools/infer/text/config.py`

By default, the inference results will be saved in `./inference_results/det_results.txt`, which can be changed via `--draw_img_save_dir` argument. 

Currently, it only supports serial inference to avoid dynamic shape issue and achieve better performance.

### Supported Algorithms and corresponding Networks 

As defined in `predict_det.py`, the supported detection algorithms are as follows. 

  | **Algorithm Name** | **Architecture** | **Support language** |  
  | :------: | :------: | :------: | 
  | DB  | dbnet_resnet50 | English |
  | DB++ | dbnetpp_resnet50 | English |
  | DB_MV3 | dbnet_mobilenetv3 | English |

### Demo Results
<p align="left">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/ce136b92-f0aa-4a05-b689-9f60d0b40db1" width=420 />
</p>

```
img_108.jpg	[[[228.0, 440.0], [403.0, 413.0], [406.0, 433.0], [231.0, 459.0]], [[282.0, 280.0], [493.0, 252.0], [499.0, 293.0], [288.0, 321.0]], [[500.0, 253.0], [636.0, 232.0], [641.0, 269.0], [505.0, 289.0]], ...]
```

<p align="left">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/61066d4a-5922-471e-b702-2ea79c3cc525" width=420 />
</p>

```
paper_sam.png	[[[1161.0, 340.0], [1277.0, 340.0], [1277.0, 378.0], [1161.0, 378.0]], [[895.0, 335.0], [1152.0, 340.0], [1152.0, 382.0], [894.0, 378.0]], ...]
```

## Text Recognition

To run text recognition on an input image or multiple images in a directory, please run:

```shell
python tools/infer/text/predict_rec.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm CRNN
```

For more argument illustrations and usage, please run `python tools/infer/text/predict_rec.py -h` or view `tools/infer/text/config.py`

By default, the inference results will be saved in `./inference_results/rec_results.txt`, which can be changed via the `--draw_img_save_dir` argument. 

Both batch-wise inference and single-mode inference are supported. 

### Supported Algorithms and corresponding Networks 

As defined in `predict_rec.py`, the supported recognition algorithms are as follows. 

  | **Algorithm Name** | **Architecture** | **Support language** |  
  | :------: | :------: | :------: | 
  | CRNN | crnn_resnet34 | English | 
  | RARE | rare_resnet34 | English |
  | CRNN_CH | crnn_resnet34_ch | Chinese |
  | RARE_CH | rare_resnet34_ch | Chinese |

Note: the above models doesn't support space char recognition.

### Demo Results

<p align="left">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/fa8c5e4e-0e05-4c93-b9a3-6e0327c1609f" width=100/>
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/8ec50bdf-ea6c-4bce-a799-2fdb8e9512b1" width=100/>
</p>

```text
word_1216.png   coffee
word_1217.png   club
```

## Text Detection and Recognition Concatenation (End2End)

To run text spoting (i.e., detect all text regions then recognize each of them) on an input image or multiple images in a directory, please run:

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} --det_algorithm DB++ --rec_algorithm CRNN
```
For more argument illustrations and usage, please run `python tools/infer/text/predict_system.py -h` or view `tools/infer/text/config.py`

By default, the inference and visualization results will be saved in `./inference_results/`, which is defined by the `--draw_img_save_dir` argument. 

### Demo Results
<p align="left">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c58fb182-32b0-4b73-b4fd-7ba393e3f397" width=420/>
</p>

```text
web_cvpr_0	[{"transcription": "canada", "points": [[430, 148], [540, 148], [540, 171], [430, 171]]}, {"transcription": "vancouver", "points": [[263, 148], [420, 148], [420, 171], [263, 171]]}, {"transcription": "cvpr", "points": [[32, 69], [251, 63], [254, 174], [35, 180]]}, {"transcription": "2023", "points": [[194, 44], [256, 45], [255, 72], [194, 70]]}, {"transcription": "june", "points": [[36, 45], [110, 44], [110, 70], [37, 71]]}, {"transcription": "1822", "points": [[114, 43], [190, 45], [190, 70], [113, 69]]}]
```
<p align="left">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=420/>
</p>

```text
img_10_0	[{"transcription": "residential", "points": [[43, 88], [149, 78], [151, 101], [44, 111]]}, {"transcription": "areas", "points": [[152, 83], [201, 81], [202, 98], [153, 100]]}, {"transcription": "when", "points": [[36, 56], [101, 56], [101, 78], [36, 78]]}, {"transcription": "you", "points": [[99, 54], [143, 52], [144, 78], [100, 80]]}, {"transcription": "pass", "points": [[140, 54], [186, 50], [188, 74], [142, 78]]}, {"transcription": "by", "points": [[182, 52], [208, 52], [208, 75], [182, 75]]}, {"transcription": "volume", "points": [[199, 30], [254, 30], [254, 46], [199, 46]]}, {"transcription": "your", "points": [[164, 28], [203, 28], [203, 46], [164, 46]]}, {"transcription": "lower", "points": [[109, 25], [162, 25], [162, 46], [109, 46]]}, {"transcription": "please", "points": [[31, 18], [109, 20], [108, 48], [30, 46]]}]
```



## How to add support for a new model inference

### Preprocessing 

The optimal preprocessing strategy can vary from model to model, especially for the resize setting (keep_ratio, padding, etc). We define the preprocessing pipeline for each model in `tools/infer/text/preprocess.py` for different tasks. 

If you find the default preprocessing pipeline or hyper-params does not meet the network requirement, please extend the if-else conditions or add a new key-value pair  the `optimal_hparam` dict, where key is the algorithm name and value is the hyper-param setting. 

### Network Inference

Supported alogirhtms and their corresponding model names (which can be checked by the `list_model()`API) are defined in the `algo_to_model_name dict in ``predict_det.py` and `predict_rec.py`. 

To add a new detection model for inference, please add a new key-value pair to `algo_to_model_name` dict, where key is the algo name parsed from `config.py` and value is the model name which is registered in `mindocr/models/{model}.py`. 

By default, model weights will be loaded from the provided pretrained URL defined in `mindocr/models/{model}.py`. If you want to load a local checkpoint instead, please set the `--det_model_dir` or `--rec_model_dir` args in command line.  

### Postproprocess

Similary, the postprocess pipeline for each algorithm can vary and is defined in `tools/infer/text/postprocess.py`.

If you find the default postprocessing pipeline or hyper-params does not meet your need, please extend the if-else conditions or add a new key-value pair  the `optimal_hparam` dict, where key is the algorithm name and value is the hyper-param setting. 
