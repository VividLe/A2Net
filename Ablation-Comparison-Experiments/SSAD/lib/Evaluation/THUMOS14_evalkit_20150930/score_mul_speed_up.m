

clear, clc

file_dir = '/disk3/zt/code/2_TIP_rebuttal/2_A2Net/output/thumos/3_CONCAT_AB_0.6/';

file_list = dir([file_dir,'*.', 'txt']);
map_all = zeros(1,length(file_list));
for i = 1:length(file_list)
    name = file_list(i).name;
    disp(name);

%     avg = 0;
    threshold = 0.5;
    %disp(i)
    [pr_all,ap_all,map]=TH14evalDet([file_dir, name],'annotation','test',threshold);
    %avg = avg + map;
    disp(map);
    map_all(i)=map;
end
[c,index] = max(map_all);
name = file_list(index).name;
sprintf('max is %s map_0.5 : %d ',name, c)

% avg = 0;
% for i=0.1:0.1:0.7
%     disp(i)
% 
%     [pr_all,ap_all,map]=TH14evalDet([file_dir, name],'annotation','test',i);
%     avg = avg + map;
% end
% disp(avg / 7);



% 
% clear, clc
% 
% file_dir = '/data/yangle/zt/ab_af_thumos_zt/output/complete_txt/';
% 
% file_list = dir(file_dir);
% map_all = zeros(1,length(file_list)-2);
% for i = 3:length(file_list)
%     name = file_list(i).name;
%     disp(name);
% 
% %     avg = 0;
%     threshold = 0.5;
%     %disp(i)
%     [pr_all,ap_all,map]=TH14evalDet([file_dir, name],'annotation','test',threshold);
%     %avg = avg + map;
%     disp(map);
%     map_all(i-2)=map;
% end
% [c,index] = max(map_all);
% name = file_list(index+2).name;
% sprintf('max is %s map_0.5 : %d ',name, c)
% 
% avg = 0;
% for i=0.1:0.1:0.7
%     disp(i)
% 
%     [pr_all,ap_all,map]=TH14evalDet([file_dir, name],'annotation','test',i);
%     avg = avg + map;
% end
% disp(avg / 7);





% clear, clc
% 
% file_dir = '/disk/yangle/ActLocSSAD_complete/output/predictions/';
% 
% file_list = dir(file_dir);
% for i = 3:length(file_list)
%     name = file_list(i).name;
%     disp(name);
% 
%     avg = 0;
%     for i=0.1:0.1:0.7
%     disp(i)
%     [pr_all,ap_all,map]=TH14evalDet([file_dir, name],'annotation','test',i);
%     avg = avg + map;
%     end
%     disp(avg / 7);
% 
% end



