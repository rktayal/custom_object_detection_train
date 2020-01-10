How to build your custom object detector?


The repo demonstrates on how to get started on creating your custom object detection model using Tensorflow.
For this demo, I am using ssd_mobilenet_v2 as the base model and will train my own class on top of it.
I have decided to train it on pedestraints using the PascalVOC2007 dataset images of person class.


By the way, here is the pedestraint detector in action


### Motivation
The purpose of writing this is twofolds:
- First, To gain better understanding of training a model using TensorFlow object detection API
- Second, The post explains all the necessary steps by performing hands-on to train an object detector with your own dataset. There are pleothera of articles, blogs written on how to perform it, but this repo not only includes the necessary dependencies,
scripts but also includes the base models and dataset to train on as well (along with the annotations),
therefore user can create a custom object detector model out of it, by performing the steps described below without bothering about the dataset. Therfore it can be a good
starting point for many people who are relatively new to Tensorflow object detection API.

At the end of the demo, you would have created your custom object detector model (trained on custom class), which 
can be used to make inferences.

### Requirements
```
Linux (Tested on CentOS 7)
Python
Python Packages
 - numpy
 - opencv-python
 - TensorFlow
```
You can use the `pip install <package_name>` command to solve the above python dependencies.

The repo has the direcotry structure as depicted below:
```
custom_object_detection_train
├─ annotations
├─ scripts
├─ images
│   ├─ test
│   └─ train
├─ pre-trained-model
├─ training
└─ README.md
```
- `annotations` folder can be used to store all `*.csv` files and respective tensorflow `*.record` files. It also contains the `label_map.pbtxt` file
which describes the mapping of class id with its name for our model.
NOTE: TF Object Detection API expects TFRecord file as input during training process
- `images` directory can contain all images in our dataset, as well as their annotation `*.xml` files. The `xml` files have the PascalVOC format.
NOTE: `images/train` is used during training, while images in `images/test` will be used to test our final model
- `pre-trained-model` has the starting checkpoint for our training job. I am using ssd_mobilenet_v2 for this demo
you can download the `*.tar.gz` file from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and extract the contents in this directory. After extracting, the contents of this direcotry should have the following files.
   - checkpoint
   - frozen_inference_graph.pb
   - model.ckpt.data-00000-of-00001
   - model.ckpt.index
   - model.ckpt.meta
   - pipeline.config
NOTE: These files will be used during weight initialization for our model.
- `training` directory will contain the training pipeline configuration file. <br />
NOTE: For our case, we will be using `pipeline.config` file of `ssd_mobilenet_v2_coco_2018_03_29`. I have copied it in `training` directory under the name `new_pipeline.config` for this case.


If you don't understand much till this point, don't worry, we will see in detail below on how we are generating the required files.

### Creating the Dataset
Tensorflow object detection API requires [TFRecord file](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) format as input, therefore we need to convert our dataset to this file format ultimately. There are other ways to get data into your model like using the `feed_dict` cmd or using the 
`tf.data.Dataset` object, but for now, we will stick to `TfRecords`. If you want to read more, this [article](https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0) covers it nicely.


For this demo, I am using PascalVOC2007 person dataset images. I have written a script `segregate.py` to segregate all the postive samples of a particular class(person in this case) from the dataset.
```
python segregate.py <path_to_annotations_dir> <path_to_img_dir> <output_dir_path>
```
After executing the above script, you should have positive samples (both `xml` & `jpg` files) of `person` class. (2008 samples)
You can divide these samples into `train` and `test` directory created above. I have put 200 samples into `test` directory. (Both the `*.jpg` & `*.xml`)
and the rest in the `train` directory.
Once we have the `*.xml` & `*.jpg` files in `train` and `test` directory, we can move to next step of converting `*.xml` to `*.csv`.
NOTE: you don't need to execute the above script again as the dataset is already sorted and organized. It is just an FYI.
#### Converting `*.xml` to `*.csv`
To do this we can write a simple script that iterates through all `*.xml` files in the `training_demo\images\train` and `training_demo\images\test` folders, and generates a `*.csv` for each of the two.
script `xml_to_csv.py` inside the `scripts` directory does the same.
```
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

# For example 
# python xml_to_csv.py -i ..\images\train -o ..\annotations\train_labels.csv
# python xml_to_csv.py -i ..\images\test -o ..\annotations\test_labels.csv
```
Once the above script is executed, there should be 2 new files under `.\annotations` folder, named `train_labels.csv` & `test_labels.csv`

#### Converting `*.csv` to `*.record`
Once we obtain our `*.csv` annotation files, we will need to convert them into TFRecord files.
`generate_tfrecord.py` under `scripts` directory converts the csv to tfrecord.


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
# python generate_tfrecord.py --label=person --csv_input=..\annotations\train_labels.csv --output_path=..\annotations\train.record --img_path=..\images\train
# python generate_tfrecord.py --label=person --csv_input=..\annotations\test_labels.csv --output_path=..\annotations\test.record --img_path=..\images\test
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
Place the file under `annotations` directory. (Already present in the repo.)

### Configuring the Training Pipeline
For this demo, we will essentially be performing transfer learning using `ssd_mobilenet_v2_coco` as the base model.
You can also choose other models listed in [Tensorflow's detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
We will be needing the configuration file as well as the latest pre-trained NN for the model we wish to use. Both can be downloaded by simply
clicking on the name of the desired model in the table found in [TensorFlow's detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)
Clicking on the name of your model should initiate a download for a `*.tar.gz` file.
Once downloaded, extract the contents into `pre-trained-model` using any decompression tool (WinZip, 7Zip) (Ignore if already done.) <br />
Now that we have downloaded and extracted, let's make changes to `*.config` file. We will be making changes to the `new_pipeline.config` under `training` directory which is identical to the `pipeline.config` extracted in the above step. You can also directory make changes to `pipeline.config` but I feel, its better to preserve the original copy for future refrence.


`new_pipeline.config` file under `training` directory has the following changes:
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
Set the PYTHONPATH variable to include the `slim` path
```
set PYTHONPATH=%PYTHONPATH%;[PATH_TO_REPO]\slim
# example set PYTHONPATH=%PYTHONPATH%;C:\Users\Z003FXH\personal\custom_object_detection_train\slim
```
Initiate the training process using the following commands
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/new_pipeline.config
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

If you observe similar logs take a moment to acknowledge yourself. You have successfully started your first training job.
As the iterations go on, `TotalLoss` will reduce, ideally it should be somewhere close to 1.
The convergence of model will depend on several hyperparameters such as `Optimizer` chosen, `learning rate`, `momentum` etc.
For now, we have chosen the default values present in the `ssd_mobilenet_v2` config pipeline for the same.
You can find good resources online on how to finetune the hyperparameters for improving the accuracy of the model and 
reducing the training time as well. Lower `TotalLoss` is better, however very low `TotalLoss` should be avoided, as the model may end up 
overfitting the dataset, meaning it will perform poorly when applied to real life data. To visually monitor the loss curve, you can have a look
at [Monitor Training Job Progress using TensorBoard](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#tensorboard-sec)
You can stop the training explicitly by pressing `ctrl+c` when `TotalLoss` comes down under 2.

On the sidenote, you will achieve decent results for such a short training time due to the fact that the detector is trained on a single class only. For more classes, total mAP won’t be as good as the one that I got and definitely longer training time would be needed to get good results. There is a also the tradeoff between model speed and model accuracy that one must consider. However that is a different story and should be fine for now.

### Exporting the trained inference graph
Once your training job is complete, you need to extract the newly trained inference graph, which can be used to perform the object detection. This can be done
by executing the following cmd:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/new_pipeline.config --trained_checkpoint_prefix training/model.ckpt-13302 --output_directory ./output_inference_graph_v1.pb
```
`input_type` value should match to what is mentioned in the pipeline configuration (`image_tensor` in this case)
`trained_checkpoint_prefix` value can be the checkpoint model generated during training under the `training` directory
Once exported, you can use the model for inferences.
Happy Training!!

### Common Issues
- [AttributeError: module 'object_detection.protos.input_reader_pb2' has no attribute 'NUMERICAL_MASKS'](https://github.com/tensorflow/models/issues/3933)
  - You need to rebuild the protobufs
- [ImportError: No module named nets](https://github.com/tensorflow/models/issues/1842) The `slim` directory should be appended to PYTHONPATH to fix it.
  - ```export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim```
- [google.protobuf.text_format.ParseError: 35:7 : Message type "object_detection.protos.SsdFeatureExtractor" has no field named "batch_norm_trainable"](https://github.com/tensorflow/models/issues/6717) Search `batch_norm_trainable: true` in your pipeline.config, then remove the line.
- ImportError: No module named object_detection  
  Set the environment variable PYTHONPATH to point to the `object_detection` directory of the cloned repo. This is typically done something like this
  - ```export PYTHONPATH=[...]/object_detection``` <br />
where [...] should be replaced with the directory where the cloned repo resides.

### References
Much of the content and scripts are taken from below link:
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
- [TensorFlow Model Zoo](https://github.com/tensorflow/models)

