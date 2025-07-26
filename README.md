# SEP-740---Project

This project consists of source code for training YOLO11 models (teachers, students, and independently trained ones). This includes 2 submodules: the standard Utralytics submodule in directory yolo_distillation/yolo_knowledge_distillation and the customized file yolo_distillation/yolo-distiller implemented by Daniel Syahputra (source: https://github.com/danielsyahputra/yolo-distiller). Due to technical problem, I have to remove .git files from both of directories in order to push this repository.

## Sub-modules:
The training script for teachers and independently trained models is located in the file yolo_distillation/yolo_knowledge_distillation/yolo_detection_training.py

The training script for distilled student models is located in the file yolo_distillation/yolo-distiller/yolo-distillation-training.py

## Requirements and Training Computer:
The requirements for Python enviroment is store in the file yolo_distillation/requirements.txt

The models were trained on a machine with 32GB ram and GPU Nvidia RTX 3050m 4GB video Ram supporting CUDA.

## Datasets:

Pet and Aquarium datasets are included in this repository.

COCO dataset, due to its enormous size, is not included in this repository. The training scripts will automatically detect and download the dataset when they are executed.
