clear, clc

threshold = 0.5;

%[pr_all,ap_all,map]=TH14evalDet('../Table1/af__remove_invalid_class/pre_ES.txt','../Table1/GT_test_dif_dur/ES','test',threshold);
%[pr_all,ap_all,map]=TH14evalDet('../Table1/pre_af/pre_S.txt','../Table1/GT_test_dif_dur/S','test',threshold);
%[pr_all,ap_all,map]=TH14evalDet('../Table1/pre_af/pre_M.txt','../Table1/GT_test_dif_dur/M','test',threshold);
%[pr_all,ap_all,map]=TH14evalDet('../Table1/pre_af/pre_L.txt','../Table1/GT_test_dif_dur/L','test',threshold);
[pr_all,ap_all,map]=TH14evalDet('../Table1/pre_af/pre_EL.txt','../Table1/GT_test_dif_dur/EL','test',threshold);

