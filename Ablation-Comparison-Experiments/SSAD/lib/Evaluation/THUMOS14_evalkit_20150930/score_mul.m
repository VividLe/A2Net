clear, clc

file_dir = '/disk/yangle/MILESTONES/ActionLocalization/output/a2net_tem/';

file_list = dir(file_dir);
for i = 3:length(file_list)
    name = file_list(i).name;
    if ~endsWith(name, '.txt')
       continue; 
    end
    disp(name);

    avg = 0;
    for i=0.5:0.1:0.5
    disp(i)
    [pr_all,ap_all,map]=TH14evalDet([file_dir, name], 'annotation', 'test', i);
    avg = avg + map;
    end
    %disp(avg / 7);

end

quit();


