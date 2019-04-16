Demonstrates on how to get started on creating your custom object detection model using Tensorflow.
For this demo, I am using ssd_mobilenet_v2 as the base model and will train my own class on top of it.
I have decided to train it on  pedestraints using my own set of images.


By the way, here is the pedestraint detector in action


The post explains all the necessary steps to train an object detector with your own dataset
using Tensorflow object detection API.


Directory Structure is depicted below:
```
training_demo
├─ annotations
├─ scripts
├─ images
│   ├─ test
│   └─ train
├─ pre-trained-model
├─ training
└─ README.md
```
- `annotations` folder can be used to store all `*.csv` files and respective tensorflow `*.record` files, containing list of annotations for our dataset images
- `images` directory can contain all images in our dataset, as well as the `*.xml` files, produced for each one using `labelImg` annotation tool
NOTE: `images/train` is used during training, while images in `images/test` will be used to test our final model
- `pre-trained-model` has the starting checkpoint for our training job. I am using ssd_mobilenet_v2 for this demo
- `training` directory contains training pipeline configuration file `*.config` as well as `*.pbtxt` label map file

### Creating the Dataset
Tensorflow object detection API requires [TFRecord file](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) format as input, therefore we need to convert our dataset to this file format ultimately.
For this demo, I am using PascalVOC2007 person dataset images. I have written a script `segregate.py` to segregate all the postive samples of a particular class(person in this case) from the dataset.
```
python segregate.py <path_to_annotations_dir> <path_to_img_dir> <output_dir_path>
```
After executing the above script, you should have positive samples (both `xml` & `jpg` files) of `person` class. (2008 samples)
You can divide these samples into `train` and `test` directory created above. I have put 200 samples into `test` directory. (Both the `*.jpg` & `*.xml`)
and the rest in the `train` directory.
Once we have the `*.xml` & `*.jpg` files in `train` and `test` directory, we can move to next step of converting `*.xml` to `*.csv`.
#### Converting `*.xml` to `*.csv`
To do this we can write a simple script that iterates through all `*.xml` files in the `training_demo\images\train` and `training_demo\images\test` folders, and generates a `*.csv` for each of the two.
script `xml_to_csv.py` does the same.
```
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

# For example
# python xml_to_csv.py -i .\images\train -o .\annotations\train_labels.csv
# python xml_to_csv.py -i .\images\test -o .\annotations\test_labels.csv
```
Once the above script is executed, there should be 2 new files under `.\annotations` folder, named `train_labels.csv` & `test_labels.csv`

#### Converting `*.csv` to `*.record`
Once we obtain our `*.csv` annotation files, we will need to convert them into TFRecord files.
`generate_tfrecord.py` converts the csv to tfrecord.


Execute the following commands:
```
# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/test
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

# For Example
# python generate_tfrecord.py --label=person --csv_input=.\annotations\train_labels.csv --output_path=.\annotations\train.record --img_path=.\images\train
# python generate_tfrecord.py --label=person --csv_input=.\annotations\test_labels.csv --output_path=.\annotations\test.record --img_path=.\images\test
```
Once done, we will have 2 new files in `annotations` folder, named `train.record` and `test.record` respectively.

### Creating the Label Map
TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

For this demo, our `label_map.pbtxt` file would look like this:
```
item {
    id: 1
    name: 'person'
}
```
Place the file under `annotations` directory

### Configuring the Training Pipeline
For this purpose of tutorial, we will essentially be performing transfer learning using `ssd_mobilenet_v2_coco` as the base model.
You can also choose other models listed in [Tensorflow's detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
We will be needing the configuration file as well as the latest pre-trained NN for the model we wish to use. Both can be downloaded by simply
clicking on the name of the desired model in the table found in [TensorFlow's detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)
Clicking on the name of your model should initiate a download for a `*.tar.gz` file.
Once downloaded, extract the contents into `pre-trained-model` using any decompression tool (WinZip, 7Zip)
Now that we have downloaded and extracted, let's make changes to `*.config` file

```
Line No. 3 num_classes: 90 needs to be changed as per your classes.
In this case num_classes: 1

Line No. 159 fine_tune_checkpoint: [PATH_TO_BE_CONFIGURED]/model.ckpt" 
In this case fine_tune_checkpoint: "pre-trained-models/model.ckpt"

(Line No. 163)Configure the paths of train_input_reader
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "annotations/train.record"	
  }
}
(Line No. 174)Similarly Configure the paths of eval_input_reader
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "annotations/test.record"
  }
}
(Line 35) batch_norm_trainable:	true
For ssd mobilenet v2, you could remove "batch_norm_trainable: true" as this field is deprecated now.

```

### Training the model
Initiate the training process using the following commands
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/pipeline.config
```

Once the training process has been initiated, you should see a series of print outs similar to the one below (plus/minus some warnings):
```
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
Please switch to tf.train.MonitoredTrainingSession
WARNING:tensorflow:From C:\Users\Z003FXH\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\contrib\slim\python\slim\learning.py:737: Supervisor.__init__ (from tensorflow.python.training.supervisor) is deprecated and will be removed in a future version.
Instructions for updating:
Please switch to tf.train.MonitoredTrainingSession
2019-04-16 19:15:35.832117: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
INFO:tensorflow:Restoring parameters from pre-trained-models/model.ckpt
INFO:tensorflow:Restoring parameters from pre-trained-models/model.ckpt
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path training/model.ckpt
INFO:tensorflow:Saving checkpoint to path training/model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:global step 1: loss = 16.3799 (73.125 sec/step)
INFO:tensorflow:global step 1: loss = 16.3799 (73.125 sec/step)
```


