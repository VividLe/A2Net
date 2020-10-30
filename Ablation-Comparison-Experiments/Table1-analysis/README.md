## Have a try

1. We provider an example results in "./Table1/example". To start with, you can copy all files under this directory and have a try
```
cp -r ./Table1/example ./
```

2. Evaluate score via "./THUMOS14_evalkit_20150930/evaluate-score.m".


## Use following steps to reprodece results in Table 1.

1. Run "GT_sort_by_action_duration.py" to re-arrange the groundtruth file, by default at directory "./GT_test_dif_dur".

2. Run "pre_postprocess_by_GT_action_duration.py" to generate the prediction files, by default at directory "./pre_af".

3. Evaluate score via "./THUMOS14_evalkit_20150930/evaluate-score.m".


## Here are two points that should be noticed:

1. For extremely short predictions, some categories contain no predictions. To smoothly run the evaluation code, we should delete these categories. In our case, we only keep followings:
```
7 BaseballPitch
9 BasketballDunk
12 Billiards
22 CliffDiving
23 CricketBowling
24 CricketShot
26 Diving
31 FrisbeeCatch
36 HammerThrow
45 JavelinThrow
68 PoleVault
79 Shotput
85 SoccerPenalty
92 TennisSwing
93 ThrowDiscus
97 VolleyballSpiking
```
You can run "remove_empty_pre_class_in_ES.py" to obtian the disposed file "./ab__remove_invalid_class/pre_ES.txt", and use it for evaluation

2. If you run "GT_sort_by_action_duration.py" multiple times, please delete the directory "GT_test_dif_dur" by yourself.

