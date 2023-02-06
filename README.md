# Federated Learning Annotated Image Repository (FLAIR): A large labelled image dataset for benchmarking in federated learning

FLAIR is a large dataset of images that captures a number of characteristics encountered in federated learning and privacy-preserving ML tasks. 
This dataset comprises approximately 430,000 images from 51,000 Flickr users, which will better reflect federated learning problems arising in practice, and it is being released to aid research in the field.

![alt text](assets/FLAIR_sample.jpeg)

## Image Labels
These images have been annotated by humans and assigned labels from a taxonomy of more than 1,600 fine-grained labels. 
All main subjects present in the images have been labeled, so images may have multiple labels. 
The taxonomy is hierarchical where the fine-grained labels can be mapped to 17 coarse-grained categories.
The dataset includes both fine-grained and coarse-grained labels so researchers can vary the complexity of a machine learning task.

## User Labels and their use for Federated Learning
We have used image metadata to extract artist names/IDs for the purposes of creating user datasets for federated learning. 
While optimization algorithms for machine learning are often designed under the assumption that each example is an independent sample from the distribution, federated learning applications deviate from this assumption in a few different ways that are reflected in our user-annotated examples. 
Different users differ in the number of images they have, as well as the number of classes represented in their image collection. 
Further, images of the same class but taken by different users are likely to have some distribution shift. 
These properties of the dataset better reflect federated learning applications, and we expect that benchmark tasks on this dataset will benefit from algorithms designed to handle such data heterogeneity.

## Getting Started
### Prerequisites
Please make sure you have python >= 3.8 and have the required packages installed (see below).
```sh
python3 -m pip install -r requirements.txt
```
### Download the dataset
Ensure you have a good network connection to download the ~6GB of image data, and enough local space to store and decompress it.
Download the dataset with the following command:
```sh
python3 download_dataset.py --dataset_dir=/path/to/data 
```
The images and metadata will be saved to the provided `dataset_dir`.
By default the script will download the down-sized images (size = 256 x 256). 
The images are split and compressed into dozens of tarball archives and will be decompressed after downloading.
If you wish to download the full-size raw images, add `--download_raw` flag in the above command.\
 ⚠️ Warning: the raw images take up to ~1.2TB disk space to store after decompressing.

After downloading and decompressing, the `dataset_dir` will have the following layout:
```
dataset_dir
├── labels_and_metadata.json      # a list of labels and metadata for each image
├── label_relationship.txt        # a list of `(fine-grained label, label)` pair
├── small_images
│   └── *.jpg                     # all down-sized images
└── raw_images                    # exists if you added `--download_raw` flag
    └── *.jpg                     # all raw images
```

### Dataset split
We include a standard train/val/test split in `labels_and_metadata.json`.
The partition is based on user ids with ratio 8:1:1, i.e. train, val and test sets have disjoint users.
Below are the numbers for each partition:

| Partition        | Train   | Val    | Test   |
| ---------------- | ------- | ------ | ------ |
| Number of users  | 41,131  | 5,141  | 5,142  |
| Number of images | 345,879 | 39,239 | 43,960 |

We recommend using the provided split for reproducible benchmarks.

### Explore the dataset
Below is an example metadata and label for one image from `labels_and_metadata.json`:
```json
{
    "user_id": "59769174@N00",
    "image_id": "14913474848",
    "fine_grained_labels": [
        "bag",
        "document",
        "furniture",
        "material",
        "printed_page"
    ],
    "labels": [
        "equipment",
        "material",
        "structure"
    ],
    "partition": "train"
}
```
Field `image_id` is the Flickr PhotoID and `user_id` is the Flickr NSID that owns the image.
Field `partition` denotes which `train/dev/test` partition the image belongs to.
Field `fine_grained_labels` is a list of annotated labels presenting the subjects in the image and `labels` is the list of coarse-grained labels obtained by  mapping fine-grained labels to higher-order categories.
The file `label_relationship.txt` includes the mapping from ~1,600 fine-grained labels to 17 higher-order categories.

We provide scripts to explore the images and labels in more detail. First you need to start a jupyter notebook:
```sh
jupyter notebook
```
- To explore the downloaded images, open in jupyter notebook [`explore_images.ipynb`](./explore_images.ipynb) which displays the images with corresponding metadata and labels.
- To explore the labels, open in jupyter notebook  [`explore_labels.ipynb`](./explore_labels.ipynb) which displays the statistics of the user and label distribution.

### (Optional) Prepare the dataset in HDF5
We provide a script to prepare the dataset in HDF5 format for more efficient processing and training:
```sh
python3 prepare_dataset.py --dataset_dir=/path/to/data --output_file=/path/to/hdf5
```
By default the script will group the images and labels by train/val/test split and then by user ids, making it suitable for federated learning experiments.
With the flag `--not_group_data_by_user`, the script will simply group the images and labels by train/val/test split and ignore the user ids, which is the typical setup for centralized training. \
⚠️ Warning: the hdf5 file take up to ~80GB disk space to store after processing.

## Benchmark FLAIR with TensorFlow Federated

### Prepare the dataset in TFRecords
We provide a script to prepare the dataset in TFRecords format for benchmarking with TensorFlow Federated:
```sh
python3 prepare_tfrecords.py --dataset_dir=/path/to/data --tfrecords_dir=/path/to/tfrecords
```
When the above script finishes, the `tfrecords_dir` will have the following layout:
```
tfrecords_dir
├── label_index.json             # a mapping from class label to index    
├── train
│   └── <user-id>.tfrecords      # tfrecords for all train users
├── dev
│   └── <user-id>.tfrecords      # tfrecords for all dev users
└── test                    
    └── <user-id>.tfrecords      # tfrecords for all test users
```
### Training in centralized setting
In centralized setting, user split is ignored and all users' data are concatenated.
Centralized model training can be done in TensorFlow Keras with the following command:
```sh
python3 -m benchmark.central_main --tfrecords_dir=/path/to/tfrecords
```
To view all available arguments, please use the following command: 
```sh
python3 -m benchmark.central_main --help
```
Please refer to our [benchmark paper](https://arxiv.org/abs/2207.08869) for the recommended hyperparameters.

### Training in federated setting
In federated setting, sampled users train on their own data locally and then share the model updates with the central server.
Federated model training can be simulated in TensorFlow Federated with the following command:
```sh
python3 -m benchmark.federated_main --tfrecords_dir=/path/to/tfrecords
```
To view all available arguments, please use the following command: 
```sh
python3 -m benchmark.federated_main --help
```
Please refer to our [benchmark paper](https://arxiv.org/abs/2207.08869) for the recommended hyperparameters.

### Training in federated setting with differential privacy
To provide a formal privacy guarantee, we use [DP-SGD](https://arxiv.org/abs/1607.00133)
in the [federated context](https://arxiv.org/abs/1710.06963) which is supported in TensorFlow Federated.
The following command enables federated learning with differential privacy:
```sh
python3 -m benchmark.federated_main --tfrecords_dir=/path/to/tfrecords --epsilon=2.0 --l2_norm_clip=0.1
```
where `epsilon` is the privacy budget and `l2_norm_clip` is the L2 norm clipping bound for Gaussian mechanism.
By default, we use [adaptive clipping](https://arxiv.org/abs/1905.03871v3) to tune the L2 norm clipping bound automatically by setting `--target_unclipped_quantile=0.1`.

### Fine-tuning a pretrained ImageNet model
Above commands are all for training from a random initialized model.
We also provide a ResNet model pretrained on ImageNet, which can be downloaded with the following command: 
```sh
wget -O /path/to/model https://docs-assets.developer.apple.com/ml-research/datasets/flair/models/resnet18.h5
```
The pretrained model is originally from [torch vision](https://download.pytorch.org/models/resnet18-f37072fd.pth) and converted to Keras format.
To use the pretrained model, please add the argument `--restore_model_path=/path/to/model` in the above training commands.

### Training a binary classifier for a single label
By default, we train a multi-label classification model where the output is a multi-hot vector indicating which labels presented in the input image.
We also provide the option to train a simpler binary classifier for a single label. 
For example, adding the argument `--binary_label=structure` trains a model only to predict whether `structure` label presented in the image. 

## Disclaimer
The annotations and Apple’s other rights in the dataset are licensed under CC-BY-NC 4.0 license. 
The images are copyright of the respective owners, the license terms of which can be found using the links provided in ATTRIBUTIONS.TXT (by matching the Image ID). 
Apple makes no representations or warranties regarding the license status of each image and you should verify the license for each image yourself.
