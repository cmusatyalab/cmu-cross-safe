# cmu-cross-safe

## Setup
Note: These instructions are based on the 
[following document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md).

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) repository so some location on the 
   server that you would like to train the model on. 
2. Install the Object Detection model by follwing [these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
    1. The `<models repository location>/research/` and `<models repository location>/research/slim` directories 
       must be in your PYTHONPATH every time you use this model. 
    2. The `<models repository location>/research/object_detection/builders/model_builder_test.py` test
       should pass after you finish the installation steps. If you run into any problems using this model,
       ensure that this test still passes.
       
## Training the Model
1. Create a directory called `experiment_data` in `<models repository location>/research/`.
2. Copy the `cross_safe_label_map.pbtxt` file from this repository into the `experiment_data` directory that you just created.
3. Copy the `create_cross_safe_tf_record.py` script from this repository into your 
   `<models repository location>/research/object_detection/dataset_tools` directory.
4. Run the following command from your `<models repository location>/research/` directory:
   ``` bash
   python object_detection/dataset_tools/create_cross_safe_tf_record.py \
       --label_map_path=experiment_data/cross_safe_label_map.pbtxt \
       --data_dir=<The location of the filtered directory from Cross-Safe>
       --output_dir=experiment_data
   ```
5. Run the following commands to download Google's COCO-pretrained Faster R-CNN with Resnet-101 model and copy it into your
   `experiment_data` directory:
   ``` bash
   wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
   tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
   cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* <models repository location>/research/experiment_data
   ```
6. Copy the `faster_rcnn_resnet101_cross_safe.config` file into your `experiment_data` directory. Edit this to configure the
   training process. In particular, you will need to specify the correct values for the `fine_tune_checkpoint`, `input_path`,
   and `label_map_path` parameters.
7. Begin the training process by running the following command from your `<models repository location>/research/` directory:
   ``` bash
   python object_detection/model_main.py \
       --pipeline_config_path=experiment_data/faster_rcnn_resnet101_cross_safe.config
       --model_dir=model_dir
       --alsologtostderr
   ```
   1. If you are running this on a server over an SSH connection, consider running this command in TMUX, so that it can
      continue running after your SSH connection is disconnected.
