# cmu-cross-safe

## Setup
Note: These instructions are based on 
[this document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md).

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) repository to some location on the 
   server that you would like to train the model on. 
2. Setup a [Python 3 Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtualenv) and activate it.
2. Install the Object Detection model by follwing [these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
    1. The `<models repository location>/research/` and `<models repository location>/research/slim` directories 
       must be in your PYTHONPATH every time you use this model. 
    2. The `<models repository location>/research/object_detection/builders/model_builder_test.py` test
       should pass after you finish the installation steps. If you run into any problems using this model,
       ensure that this test still passes.
       
## Training the Model
1. Create a directory called `experiment_data` in `<models repository location>/research`.
2. Copy the `cross_safe_label_map.pbtxt` file from this repository into the `experiment_data` directory that you just created.
3. Copy the `create_cross_safe_tf_record.py` script from this repository into your 
   `<models repository location>/research/object_detection/dataset_tools` directory.
4. Run the following command from your `<models repository location>/research` directory:
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
6. Copy the `faster_rcnn_resnet101_cross_safe.config` file from this repository into your `experiment_data` directory. Edit 
   this to configure the
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
8. Check on the status of your training by running `tensorboard --logdir=model_dir` from your 
   `<models repository location>/research/` directory.    
   1. This should be done from another shell while the training process is being run. Ensure that your virtualenv is 
      activated from this shell as well.
   2. Open a web browser and navigate to the location outputted by the Tensorboard command. Note that it may take some 
      time for results to be available in Tensorboard.
9. Once you are ready to stop training, hold control and press c in the shell that is running the training process. In my
   experience, it took a few hours to train this classifier.
   
## Exporting the classifier
1. List the files in the `<models repository location>/research/model_dir` directory. Identify the number of the checkpoint   
   that you would like to export. I typically pick the checkpoint with the highest number.
2. Run the following command from the `<models repository location>/research/` directory:
   ``` bash
   python object_detection/export_inference_graph.py \
       --input_type image_tensor \
       --pipeline_config_path experiment_data/faster_rcnn_resnet101_cross_safe.config \
       --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
       --output_directory exported_graphs
   ```
3. The classifier will be written to the `exported_graphs` directory.
4. Nvidia only offers an old version of TensorFlow for the Jetson TX2. If you want to run a classifier on the Jetson board, 
   a graph exported from a current version of Tensorflow will not load properly in the old version for the Jetson. You can 
   get around this problem by copying the checkpoint files over to the Jetson and running the
   `object_detection/export_inference_graph.py` command on the Jetson board directly. 
   
## Evaluating the classifier
1. Copy the `compute_metrics.py` script from this repository into your `<models repository location>/research/` directory.
2. Edit the copy of this script and update the location of the frozen inference graph, the label map location, the location
   of the pickle file with the list of files in the hold out set, and the location of the JPEG files from the filtered 
   directory from Cross-Safe.
3. Run the script. It will print out information about the classifier's performance and it will create the pickle files:
   `dont_walk_labels.p`, `dont_walk_scores.p`, `walk_labels.p`, and `walk_scores.p`.
4. Checkout this repository on a computer with a graphical user interface (or SSH into a server with X Forwarding enabled).
5. Move the four pickle files created by `compute_metrics.py` into the `precision_recall_curves` directory in this   
   repository on the computer with access to a GUI. 
5. Navigate to the `precision_recall_curves` directory with a shell. 
6. Add the dependencies listed in `requirements.txt` to 
   [a virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/#using-requirements-files).
7. Run the `plot_precision_recall.py` script to generate the precision-recall curve.
