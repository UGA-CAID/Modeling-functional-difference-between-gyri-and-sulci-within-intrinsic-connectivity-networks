clear;
clc;
close all;

addpath /data/hzb1/Projects/S900_RSN/Analyses/fft_analysis
addpath /data/hzb1/matlab_toolbox/color

save_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/fft_corr_calc/';
if ~exist(save_path,'dir')
    mkdir(save_path);
end

result_root_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1/';
save_orig_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Group_1/';

%% multi group comparision
group1_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1/';
group2_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_2/';
taskshuffled1_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Data_taskshuffled_Group_1_test_consistent/';
taskshuffled2_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Data_taskshuffled_Group_2_test_consistent/';

group1_gs_p_corr = load([save_path, 'group1_gs_p_corr.mat']);
group1_gs_p_corr = group1_gs_p_corr.group1_gs_p_corr;
group2_gs_p_corr = load([save_path, 'group2_gs_p_corr.mat']);
group2_gs_p_corr = group2_gs_p_corr.group2_gs_p_corr;
taskshuffled1_gs_p_corr = load([save_path, 'taskshuffled1_gs_p_corr.mat']);
taskshuffled1_gs_p_corr = taskshuffled1_gs_p_corr.taskshuffled1_gs_p_corr;
taskshuffled2_gs_p_corr = load([save_path, 'taskshuffled2_gs_p_corr.mat']);
taskshuffled2_gs_p_corr = taskshuffled2_gs_p_corr.taskshuffled2_gs_p_corr;

cmatrix = taskshuffled1_gs_p_corr;

cmatrix(:,2) = [];
cmatrix(2,:) = [];
cmatrix(4,:) = [];
[h1,p1] = ttest2(cmatrix(:,4),cmatrix(:,1),'Tail','left');
[h2,p2] = ttest2(cmatrix(:,4),cmatrix(:,2),'Tail','left');
[h3,p3] = ttest2(cmatrix(:,4),cmatrix(:,3),'Tail','left');
[h4,p4] = ttest2(cmatrix(:,4),cmatrix(:,5),'Tail','left');
[h5,p5] = ttest2(cmatrix(:,4),cmatrix(:,6),'Tail','left');
[h6,p6] = ttest2(cmatrix(:,4),cmatrix(:,7),'Tail','left');
[h7,p7] = ttest2(cmatrix(:,4),cmatrix(:,8),'Tail','left');


% [group1_p,group1_h,group1_stats] = anova1(group1_gs_p_corr);
% [group1_c,group1_m,group1_h,group1_nms] = multcompare(group1_stats);
% group1_gs_p_corr(:,2) = [];
% group1_gs_p_corr(2,:) = [];
% group1_gs_p_corr(4,:) = [];
% [group1_p,group1_h,group1_stats] = anova1(group1_gs_p_corr);
% [group1_c,group1_m,group1_h,group1_nms] = multcompare(group1_stats);

% [group2_p,group2_h,group2_stats] = anova1(group2_gs_p_corr);
% [group2_c,group1_m,group2_h,group2_nms] = multcompare(group2_stats);
% group2_gs_p_corr(:,2) = [];
% group2_gs_p_corr(2,:) = [];
% group2_gs_p_corr(4,:) = [];
% [group2_p,group2_h,group2_stats] = anova1(group2_gs_p_corr);
% [group2_c,group1_m,group2_h,group2_nms] = multcompare(group2_stats);

% [taskshuffled1_p,taskshuffled1_h,taskshuffled1_stats] = anova1(taskshuffled1_gs_p_corr);
% [taskshuffled1_c,taskshuffled1_m,taskshuffled1_h,taskshuffled1_nms] = multcompare(taskshuffled1_stats);
% taskshuffled1_gs_p_corr(:,2) = [];
% taskshuffled1_gs_p_corr(2,:) = [];
% taskshuffled1_gs_p_corr(4,:) = [];
% [taskshuffled1_p,taskshuffled1_h,taskshuffled1_stats] = anova1(taskshuffled1_gs_p_corr);
% [taskshuffled1_c,taskshuffled1_m,taskshuffled1_h,taskshuffled1_nms] = multcompare(taskshuffled1_stats);

% [taskshuffled2_p,taskshuffled2_h,taskshuffled2_stats] = anova1(taskshuffled2_gs_p_corr);
% [taskshuffled2_c,taskshuffled2_m,taskshuffled2_h,group2_nms] = multcompare(taskshuffled2_stats);
% taskshuffled2_gs_p_corr(:,2) = [];
% taskshuffled2_gs_p_corr(2,:) = [];
% taskshuffled2_gs_p_corr(4,:) = [];
% [taskshuffled2_p,taskshuffled2_h,taskshuffled2_stats] = anova1(taskshuffled2_gs_p_corr);
% [taskshuffled2_c,taskshuffled2_m,taskshuffled2_h,group2_nms] = multcompare(taskshuffled2_stats);