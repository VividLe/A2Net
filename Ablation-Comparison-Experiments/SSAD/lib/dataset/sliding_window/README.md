## Extract Feature

### Extract frames and calculate optical flow
We use [TSN](https://github.com/yjxiong/temporal-segment-networks#extract-frames-and-optical-flow-images) to extract frames and calculate optical flow. Here are some important points:

1. We use CV2 to extract frames from raw input videos.
2. We adopt [TVL1](https://pequan.lip6.fr/~bereziat/cours/master/vision/papers/zach07.pdf) algothrim to calculate optical flow.
3. For each frame, we obtain 3 files, e.g., flow_x_00001.jpg, flow_y_00001.jpg, img_00001.jpg. The last frame of the video is discarded. Thus, The number of images is equal to the number of flow frames.


#### THUMOS14 dataset
1. val set frame rate.

  | frame rate (fps)  | video numbers |
  | ------------- | ------------- |
  | 30 | 187 |
  | 29.97 | 7 |
  | 25 | 6 |

Video names with 25 fps:
```
video_validation_0000311, video_validation_0000420, video_validation_0000666, video_validation_0000419, video_validation_0000484, video_validation_0000413
```

2. Test set frame rate.

  | frame rate (fps)  | video numbers |
  | ------------- | ------------- |
  | 30 | 196 |
  | 29.97 | 11 |
  | 25 | 5 |
  | 24 | 1 |

Video names with 25 fps:
```
video_test_0000950, video_test_0001255, video_test_0001459, video_test_0001058, video_test_0001195
```
Video names with 24 fps:
```
video_test_0001207
```

3. Annotation label
In both the val and test dataset, all actions belong to "CliffDiving" are also labelled as action "Diving", which is a bit conflict. Thus, for evaluation, we assign all actions detected as "CliffDiving" to "Diving" as well.


#### ActivityNet 1.3 dataset
1. There are 14950 videos.

- Training dataset: 10023 videos. The video "1v5HE_Nm99g" is available, but not correct, discarded it.
- Validation dataset: 4926 videos. All available.
- Teting dataset: 5044 videos. All available.


#### HACS dataset
1. 


### Extract I3D feature
We use [pytorch-i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction) to extract I3D feature for video data.

1. The official I3D model is [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d), the corresponding paper is [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750). Official I3D model is based on TensorFlow, the parameters of [pytorch-i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction) is converted from the original weight file.
2. Data processing. We follow [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) to carry on data processing. 

RGB: (1) Resize frames to 224x224.
     (2) Rescale pixel value between -1 and 1.
Flow: (1) Extract flow using TVL1 algorithm.
      (2) Rescale pixel value between -1 and 1. (Different from official Kinetics-I3D operation, (a) truncate pixel values to [-20, 20]. (b) rescale to [-1, 1])

3. Other I3D resources: [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).


### Generate sliding window ground truth information
As for THUMOS14, we adopt sliding window input data.
1. The sliding windows contain 128 features. Features are calculated with stride=4, thus one sliding window corresponding to 512 frames

