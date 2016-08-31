# Dataset examples #

In each folder of this directory, there are only three or four pictures on display. They are typical pictures selected from a large dataset. For simplicity, we use the black sponge as our object for training and testing. 

## The brief description of each folder ##

`pic_yes`: Training and validation data, consisting of 20 classes of grasping pictures in 20 subfolders. For each subfolder, there are 250 pictures.

`pic_no`: Training and validation data, consisting of 5000 non-grasping pictures, which are also divided into 20 subfolders.

`pic_yes_test`: Test data. 100 pictures for successful grasp (obsolete).

`pic_yes_test2`: Test data. 100 pictures for successful grasp (obsolete).

`pic_no_test`: Test data. 100 non-grasping pictures.

`pic_yes_test0716`： Test data. 20 classes of grasping pictures, each of which has 5 postures. There are totally 100 pictures.

(In our experiment presented in the paper, we use `pic_yes` and `pic_no` as our training and validation set. Meanwhile, we use `pic_yes_test0716` and `pic_no_test` as our test set.)