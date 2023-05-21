# MindOCR Online Inference

## Text Detection

To run text detection on an input image or multiple images in a directory, please run:

```shell
$ python tools/infer/text/predict_det.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm DB++
```

For more argument illustrations and usage, please run `python tools/infer/text/predict_det.py -h` or view `tools/infer/text/config.py`

By default, the inference results will be saved in `./inference_results/det_results.txt`, which is defined by the `--draw_img_save_dir` argument. 

Example inference result:
```text
img_101.jpg [[[829.0, 110.0], [1000.0, 76.0], [1017.0, 161.0], [846.0, 196.0]]]
```

Currently, it only supports serial inference to avoid dynamic shape issue and achieve better performance.

### Supported Algorithms and corresponding Networks 

Algorithm name, network name, language 

- DB, dbnet_resnet50, English 
- DB++, dbnetpp_resnet50, English

## Text Recognition

To run text recognition on an input image or multiple images in a directory, please run:

```shell
$ python tools/infer/text/predict_rec.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm CRNN
```

For more argument illustrations and usage, please run `python tools/infer/text/predict_rec.py -h` or view `tools/infer/text/config.py`

By default, the inference results will be saved in `./inference_results/rec_results.txt`, which is defined by the `--draw_img_save_dir` argument. 

Example inference result:
```text
word_1200.png   dior
word_1201.png   stage
```

Both batch-wise inference and single-mode inference are supported. 

### Supported Algorithms and corresponding Networks 

Algorithm name, network name, language 

- CRNN, crnn_resnet34, English 
- RARE, rare_resnet34, English
- CRNN_CH, crnn_resnet34_ch, Chinese
- RARE_CH, rare_resnet34_ch, Chinese


## Text Detection and Recognition Concatenation (End2End)

To run text spoting (i.e., detect all text regions then recognize each of them) on an input image or multiple images in a directory, please run:

```shell
$ python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} --det_algorithm DB++ --rec_algorithm CRNN
```
For more argument illustrations and usage, please run `python tools/infer/text/predict_system.py -h` or view `tools/infer/text/config.py`

By default, the inference and visualization results will be saved in `./inference_results/`, which is defined by the `--draw_img_save_dir` argument. 

Example inference results:
```text
img_101_3   [{"transcription": "sale", "points": [[824, 104], [1001, 75], [1016, 168], [839, 197]]}]
```

===> TODO: Add visualziation here, 

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
