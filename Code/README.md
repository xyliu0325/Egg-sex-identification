# Colorimetric Image Analysis and Swin Transformer Classification

A computational workflow for extracting RGB responses from colorimetric spot arrays, constructing color-difference images, and performing female–male image classification with a Swin Transformer.

The implementation connects spatially defined color measurements with image-based learning and quantitative evaluation. Separate modules support inspection of the RGB extraction procedure, image construction, model optimization, and prediction results. Example images are provided in `examples/`.

## Computational workflow

1. **RGB extraction** — Measure the central region of each spot in a 3 × 6 array, apply median absolute deviation filtering, and export mean channel intensities with visual inspection overlays.
2. **Color-difference image construction** — Calculate channel-wise differences between paired RGB measurements and encode each group of 18 responses as a colorimetric image.
3. **Class organization** — Organize images into `Female` and `Male` folders using predefined sample identifiers.
4. **Model training** — Optimize a Swin-Base classifier and record training and validation loss, accuracy, and learning rate.
5. **Model evaluation** — Generate prediction records, classification reports, confusion matrices, receiver operating characteristic curves, and precision–recall curves.

## Repository structure

```text
.
├── data_processing/
│   ├── extract_rgb.py
│   ├── generate_color_diff.py
│   └── sort_by_sex.py
├── swin_transformer/
│   ├── model.py
│   ├── my_dataset.py
│   ├── utilsnew.py
│   ├── train.py
│   ├── predict.py
│   ├── create_confusion_matrix.py
│   ├── select_incorrect_samples.py
│   └── class_indices.json
├── examples/
│   ├── Female/
│   └── Male/
├── environment_info.txt
├── requirements.txt
└── README.md
```

| Module | Function |
| --- | --- |
| `extract_rgb.py` | Spatial calibration, circular region sampling, robust pixel filtering, and RGB export |
| `generate_color_diff.py` | Channel-wise difference calculation and 18-spot image generation |
| `sort_by_sex.py` | Image organization according to predefined female and male identifiers |
| `model.py` | Swin Transformer architecture and model constructors |
| `my_dataset.py` | Image loading and batch collation |
| `utilsnew.py` | Class-wise image splitting, training, validation, and metric visualization |
| `train.py` | Model optimization, checkpoint saving, and training logs |
| `predict.py` | Single-image inference and class probability visualization |
| `create_confusion_matrix.py` | Quantitative evaluation and export of prediction results |
| `select_incorrect_samples.py` | Identification and recording of misclassified validation images |

## Experimental environment

The author-reported experimental environment is summarized below.

| Component | Version |
| --- | --- |
| Python | 3.8.19 |
| PyTorch | 2.4.1+cu118 |
| CUDA reported by PyTorch | 11.8 |
| cuDNN reported by PyTorch | 90100 |

From the repository root, install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The dependency list is not version-locked. The table above records the confirmed experimental versions; `environment_info.txt` retains the corresponding environment details.

## RGB extraction

Run the extraction script from the repository root, specifying a folder containing spot-array images:

```bash
python data_processing/extract_rgb.py --input "path/to/spot_images" --calibrate
```

During calibration, select the centers of the upper-left, upper-right, lower-left, and lower-right spots, in that order. Right-click to undo a selection and press Enter to confirm. Calibration should be performed separately for images with different cropping or orientation.

To reuse the saved spatial calibration:

```bash
python data_processing/extract_rgb.py --input "path/to/spot_images" --calibration "path/to/spot_images/RGB_results_MAD35/calibration.json"
```

The sampling radius defaults to 0.13 times the minimum adjacent-center spacing and can be adjusted with `--radius-fraction`.

Within each circular sampling region, the script estimates the channel-wise median and median absolute deviation (MAD). A pixel is retained only when all three channels satisfy:

```text
|I_c - median_c| <= 3.5 × max(1.4826 × MAD_c, 1.0)
```

Filtering is performed once on the original 8-bit pixel values. Mean RGB intensities are calculated from retained pixels and exported to two decimal places. Spot indices follow row-major order, from left to right and top to bottom.

Each processed folder receives a `RGB_results_MAD35/` directory containing:

| Output | Description |
| --- | --- |
| `RGB_results.xlsx` | Image names, spot indices, and filtered mean RGB intensities |
| `calibration.json` | Normalized corner coordinates and extraction parameters |
| `previews/` | Side-by-side original and annotated images; red outlines mark sampling regions and magenta marks rejected pixels |
| `errors.txt` | Image-processing error log |

## Color-difference image construction

`generate_color_diff.py` reads paired Excel workbooks containing `Blue`, `Green`, and `Red` columns. Corresponding rows must represent the same sample and spot in the same order. For each channel, the script calculates:

```text
D = |C_A - C_B|
T = 17 × |D - 4|
```

Both representations are exported, and the selected representation is rendered as groups of 18 circular spots on a black background. The default output dimensions are 4000 × 3000 pixels.

Before execution, configure the input and output paths in the script, create the destination folders, and adjust both fixed row limits in `mapProcess` to match the intended number of measurements. Then run:

```bash
python data_processing/generate_color_diff.py
```

For class organization, configure `folder_path`, `female_prefix`, and `male_prefix` in `sort_by_sex.py`, then run:

```bash
python data_processing/sort_by_sex.py
```

The script copies matching images into `Female` and `Male` subfolders and skips existing destination files.

## Swin Transformer training

Run the model scripts from `swin_transformer/`, where relative paths for class mappings, logs, and checkpoints are resolved:

```bash
cd swin_transformer
python train.py --data-path "path/to/class_images" --weights "path/to/pretrained_checkpoint.pth" --num_classes 2 --epochs 100 --batch-size 8 --lr 0.000001 --device cuda:0
```

The image root supplied through `--data-path` should contain the `Female` and `Male` class folders. The training loader reads the pretrained checkpoint through its `model` entry and removes classification-head parameters before loading the backbone.

The following values describe the current implementation defaults.

| Setting | Default |
| --- | --- |
| Model constructor | `swin_base_patch4_window7_224_in22k` |
| Classification output | 2 classes |
| Input crop | 224 × 224 pixels |
| Optimizer | AdamW |
| Learning rate | 0.000001 |
| Weight decay | 0.05 |
| Training epochs | 100 |
| Batch size | 8 |
| Validation sampling fraction | 0.30 within each class, rounded down to an integer count |
| Random seed for image splitting | 999 |

Training and validation images are resized, center-cropped, and normalized with channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The split seed governs image selection; it does not seed all stochastic operations during optimization.

Training writes epoch checkpoints and `final_model.pth` to `weights/`. The `logs/` directory contains `training_log.csv`, `training_metrics.png`, and TensorBoard event files under `tensorboard/`.

## Evaluation and inference

From `swin_transformer/`, ensure the output directory exists and evaluate a selected trained checkpoint:

```bash
python -c "from pathlib import Path; Path('logs').mkdir(exist_ok=True)"
python create_confusion_matrix.py --data-path "path/to/class_images" --weights "path/to/trained_checkpoint.pth" --num_classes 2 --batch-size 8 --device cuda:0
```

Evaluation uses the validation subset returned by the same splitting function. It exports `predictions.csv`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`, `roc_curve_data.xlsx`, and `pr_curve.png` under `logs/`.

The supplied class mapping is `0: Female` and `1: Male`. ROC and precision–recall calculations use the predicted probability of class 1. The splitting function regenerates `class_indices.json` from alphabetically ordered class-folder names; retain the mapping associated with the trained checkpoint for inference.

To record misclassified validation images:

```bash
python select_incorrect_samples.py --data-path "path/to/class_images" --weights "path/to/trained_checkpoint.pth" --num_classes 2 --device cuda:0
```

The resulting `record.txt` lists image paths, reference labels, and predicted labels.

For single-image inference, set `img_path` and `model_weight_path` in `predict.py`, then run:

```bash
python predict.py
```

The script displays the image with its predicted class and prints the class probabilities.

