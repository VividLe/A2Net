clear, clc

avg = 0;
for i=0.1:0.1:0.7
  disp(i)
  [pr_all,ap_all,map]=TH14evalDet('/disk3/zt/code/2_TIP_rebuttal/2_A2Net/output/thumos/output_toy/action_detection_00.txt','annotation','test',i);
  avg = avg + map;
end

disp(avg / 7);
