# Practice-of-JSRG
>  A Systematic Practice of Judging the Success of a Robotic Grasp Using Convolutional Neural Network

##Brief description of the Practice
  In order to collect pictures, we construct a data acquisition platform capable of robot arm grasping and photo capturing.
All of the instructions packed and sent to robot arm are done by some specific python scripts.

  After collecting enough image data, we converted raw images and related labels into some binary files which is proposed by tensorflow(an open source software framework made by Google).

  Finally we constructed our nets base on tensorflow and develop a set of programs for network training and testing.

##Brief description of the composition
  All of the python scripts are located in root directory of our project. And raw images are located in data directory.(But we only provide a little raw images for you in github)

  In converted_data directory, some files with the suffix 'tfrecords' are converted by raw images. And those data files are inputed into Network.

  Some information recorded by tensorflow are stored with specific files in the folders named with saver and summary.

  In addition, a lot of valuable information you want to record are in log directory with log file.

##Brief description of the Code
  We divide all of the python scripts into three absolute sections depend on their function.

* Image acquisition: camera.py, instructions.py, randcontrl.py, socketLink.py

* Image convertion: convert_image_tf.py

* Network train and test: tf_inputs.py, Robot_data.py, RobotNet.py, Robot_Train_Test.py, Nets.py

<br/>Attention: Before you run the scripts, you need to configure the Global Variable in the begin of the scripts. 


<br/>Please contact to me for more information. Email address:767924520@qq.com
