clear;
clc;
close all;

load('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Group_1/ratiomatrix.mat')
x_tmp = [ratiomatrix];
load('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Group_2/ratiomatrix.mat')
x_tmp = [x_tmp; ratiomatrix];
load('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Data_taskshuffled_Group_1_test_consistent/ratiomatrix.mat')
x_tmp = [x_tmp; ratiomatrix];
load('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Data_taskshuffled_Group_2_test_consistent/ratiomatrix.mat')
x_tmp = [x_tmp; ratiomatrix];

% x = [x_tmp(1,:); x_tmp(3,:); x_tmp(2,:); x_tmp(4,:)];
% x(:,5) = [];
% [p,table,stats] = anova2(x,2)
x = [x_tmp(1,:); x_tmp(3,:); x_tmp(5,:); x_tmp(7,:); x_tmp(2,:); x_tmp(4,:); x_tmp(6,:); x_tmp(8,:)];
x(:,5) = [];

%% gyri and sulci analysis
[p,table,stats] = anova2(x,4)
% 
% 
% group analysis
x_gyri = x(1:4,:);
[p_gyri,table,stats] = anova2(x_gyri,1)
x_sulci = x(5:8,:);
[p_sulci,table,stats] = anova2(x_sulci,1)

x_sulci = [x(6,:);x(8,:)];
[p_sulci,table,stats] = anova2(x_sulci,1)