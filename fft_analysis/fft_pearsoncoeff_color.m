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

%% compute fft
ratiomatrix = zeros(2,10);
for rsnid = 1 : 10
    if rsnid == 5 
        continue
    end
    
    disp(rsnid)
    result_path = [result_root_path, num2str(rsnid, '%02d'),'/'];

    %% Conv_pred
    conv_pred = load([result_path, 'conv_pred.mat']);
    x = conv_pred.conv_pred;
    x_gyri = x(1,:,:);
    x_gyri = squeeze(x_gyri)';
    group1_x_sulci = x(2,:,:);
    group1_x_sulci = squeeze(group1_x_sulci)';
    num_filter = size(x,2);
    
     %% compute average fft for filter 0 excluded
    % gyri
    f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;       
        P1_sum = P1_sum + P1;
    end
    gyri_amp = P1_sum/pcount;
    
    % sulci
    f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = group1_x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    sulci_amp = P1_sum/pcount;
    
    p_corr = corrcoef(sulci_amp, gyri_amp)
end

%% multi group comparision
group1_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1/';
group2_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_2/';
taskshuffled1_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Data_taskshuffled_Group_1_test_consistent/';
taskshuffled2_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Data_taskshuffled_Group_2_test_consistent/';

group1_gyri = [];
group1_sulci = [];
group2_gyri = [];
group2_sulci = [];
taskshuffled1_gyri = [];
taskshuffled1_sulci = [];
taskshuffled2_gyri = [];
taskshuffled2_sulci = [];
group1_gs = [];
group2_gs = [];
taskshuffled1_gs = [];
taskshuffled2_gs = [];

for rsnid = 1 : 10
    if rsnid == 5 
        continue
    end
    
    disp(rsnid)
    group1_result_path = [group1_path, num2str(rsnid, '%02d'),'/'];
    group2_result_path = [group2_path, num2str(rsnid, '%02d'),'/'];
    taskshuffled1_result_path = [taskshuffled1_path, num2str(rsnid, '%02d'),'/'];
    taskshuffled2_result_path = [taskshuffled2_path, num2str(rsnid, '%02d'),'/'];

    %% GROUP 1
    %% group1_Conv_pred
    conv_pred = load([group1_result_path, 'conv_pred.mat']);
    group1_x = conv_pred.conv_pred;
    group1_x_gyri = group1_x(1,:,:);
    group1_x_gyri = squeeze(group1_x_gyri)';
    group1_x_sulci = group1_x(2,:,:);
    group1_x_sulci = squeeze(group1_x_sulci)';
    group1_num_filter = size(group1_x,2);
    
    %% compute average fft for filter 0 excluded
    % gyri
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = group1_x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;       
        P1_sum = P1_sum + P1;
    end
    group1_gyri_amp = P1_sum/pcount;
    group1_gyri = [group1_gyri group1_gyri_amp];
    
    % sulci
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = group1_x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    group1_sulci_amp = P1_sum/pcount;
    group1_sulci = [group1_sulci group1_sulci_amp];
    
    group1_GS = group1_sulci_amp./group1_gyri_amp;
    group1_gs = [group1_gs group1_GS];
    
    %% GROUP 2
    %% group1_Conv_pred
    conv_pred = load([group2_result_path, 'conv_pred.mat']);
    group2_x = conv_pred.conv_pred;
    group2_x_gyri = group2_x(1,:,:);
    group2_x_gyri = squeeze(group2_x_gyri)';
    group2_x_sulci = group2_x(2,:,:);
    group2_x_sulci = squeeze(group2_x_sulci)';
    group2_num_filter = size(group2_x,2);
    
    %% compute average fft for filter 0 excluded
    % gyri
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = group2_x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;       
        P1_sum = P1_sum + P1;
    end
    group2_gyri_amp = P1_sum/pcount;
    group2_gyri = [group2_gyri group2_gyri_amp];
    
    % sulci
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = group2_x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    group2_sulci_amp = P1_sum/pcount;
    group2_sulci = [group2_sulci group2_sulci_amp];
    
    group2_GS = group2_sulci_amp./group2_gyri_amp;
    group2_gs = [group2_gs group2_GS];
    
    %% taskshuffled 1
    %% group1_Conv_pred
    conv_pred = load([taskshuffled1_result_path, 'conv_pred.mat']);
    taskshuffled1_x = conv_pred.conv_pred;
    taskshuffled1_x_gyri = taskshuffled1_x(1,:,:);
    taskshuffled1_x_gyri = squeeze(taskshuffled1_x_gyri)';
    taskshuffled1_x_sulci = taskshuffled1_x(2,:,:);
    taskshuffled1_x_sulci = squeeze(taskshuffled1_x_sulci)';
    taskshuffled1_num_filter = size(taskshuffled1_x,2);
    
    %%compute average fft for filter 0 excluded
    % gyri
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = taskshuffled1_x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;       
        P1_sum = P1_sum + P1;
    end
    taskshuffled1_gyri_amp = P1_sum/pcount;
    taskshuffled1_gyri = [taskshuffled1_gyri taskshuffled1_gyri_amp];
    
    % sulci
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = taskshuffled1_x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    taskshuffled1_sulci_amp = P1_sum/pcount;
    taskshuffled1_sulci = [taskshuffled1_sulci taskshuffled1_sulci_amp];
    
    taskshuffled1_GS = taskshuffled1_sulci_amp./taskshuffled1_gyri_amp;
    taskshuffled1_gs = [taskshuffled1_gs taskshuffled1_GS];
    
    %% Taskshuffled 2
    %% group1_Conv_pred
    conv_pred = load([taskshuffled2_result_path, 'conv_pred.mat']);
    taskshuffled2_x = conv_pred.conv_pred;
    taskshuffled2_x_gyri = taskshuffled2_x(1,:,:);
    taskshuffled2_x_gyri = squeeze(taskshuffled2_x_gyri)';
    taskshuffled2_x_sulci = taskshuffled2_x(2,:,:);
    taskshuffled2_x_sulci = squeeze(taskshuffled2_x_sulci)';
    taskshuffled2_num_filter = size(taskshuffled2_x,2);
    
    %% compute average fft for filter 0 excluded
    % gyri
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = taskshuffled2_x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;       
        P1_sum = P1_sum + P1;
    end
    taskshuffled2_gyri_amp = P1_sum/pcount;
    taskshuffled2_gyri = [taskshuffled2_gyri_amp taskshuffled2_gyri];
    
    % sulci
%     f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = taskshuffled2_x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    taskshuffled2_sulci_amp = P1_sum/pcount;
    taskshuffled2_sulci = [taskshuffled2_sulci taskshuffled2_sulci_amp];
    
    taskshuffled2_GS = taskshuffled2_sulci_amp./taskshuffled2_gyri_amp;
    taskshuffled2_gs = [taskshuffled2_gs taskshuffled2_GS];
    
%     sulci_matrix = [group1_sulci_amp group2_sulci_amp taskshuffled1_sulci_amp taskshuffled2_sulci_amp];
%     gyri_matrix = [group1_gyri_amp group2_gyri_amp taskshuffled1_gyri_amp taskshuffled2_gyri_amp];
%     gs_matrix = [group1_GS group2_GS taskshuffled1_GS taskshuffled2_GS];
    
%     sulci_p_corr = corrcoef(sulci_matrix)
%     gyri_p_corr = corrcoef(gyri_matrix)
%     gs_p_corr = corrcoef(gs_matrix)
end
group1_sulci_p_corr = corrcoef(group1_sulci)
% heatmap
imagesc(group1_sulci_p_corr, [0.75 1]);
colorbar
colormap parula;
save([save_path 'group1_sulci_p_corr.mat'], 'group1_sulci_p_corr');
saveas(gcf, [save_path 'group1_sulci_p_corr_color.png']);
clf;

group1_gyri_p_corr = corrcoef(group1_gyri)
% heatmap
imagesc(group1_gyri_p_corr, [0.9 1]);
colorbar
colormap parula;
save([save_path 'group1_gyri_p_corr.mat'], 'group1_gyri_p_corr');
saveas(gcf, [save_path 'group1_gyri_p_corr_thre_adjusted_color.png']);

group2_sulci_p_corr = corrcoef(group2_sulci)
% heatmap
imagesc(group2_sulci_p_corr, [0.75 1]);
colorbar
colormap parula;
save([save_path 'group2_sulci_p_corr.mat'], 'group2_sulci_p_corr');
saveas(gcf, [save_path 'group2_sulci_p_corr_color.png']);

group2_gyri_p_corr = corrcoef(group2_gyri)
% heatmap
imagesc(group2_gyri_p_corr, [0.9 1]);
colorbar
colormap parula;
save([save_path 'group2_gyri_p_corr.mat'], 'group2_gyri_p_corr');
saveas(gcf, [save_path 'group2_gyri_p_corr_thre_adjusted_color.png']);

taskshuffled1_sulci_p_corr = corrcoef(taskshuffled1_sulci)
% heatmap
imagesc(taskshuffled1_sulci_p_corr, [0.75 1]);
colorbar
colormap parula;
save([save_path 'taskshuffled1_sulci_p_corr.mat'], 'taskshuffled1_sulci_p_corr');
saveas(gcf, [save_path 'taskshuffled1_sulci_p_corr_color.png']);

taskshuffled1_gyri_p_corr = corrcoef(taskshuffled1_gyri)
% heatmap
imagesc(taskshuffled1_gyri_p_corr, [0.9 1]);
colorbar
colormap parula;
save([save_path 'taskshuffled1_gyri_p_corr.mat'], 'taskshuffled1_gyri_p_corr');
saveas(gcf, [save_path 'taskshuffled1_gyri_p_corr_adjusted_color.png']);


taskshuffled2_sulci_p_corr = corrcoef(taskshuffled2_sulci)
% heatmap
imagesc(taskshuffled2_sulci_p_corr, [0.75 1]);
colorbar
colormap parula;
save([save_path 'taskshuffled2_sulci_p_corr.mat'], 'taskshuffled2_sulci_p_corr');
saveas(gcf, [save_path 'taskshuffled2_sulci_p_corr_color.png']);


taskshuffled2_gyri_p_corr = corrcoef(taskshuffled2_gyri)
% heatmap
imagesc(taskshuffled2_gyri_p_corr, [0.9 1]);
colorbar
colormap parula;
save([save_path 'taskshuffled2_gyri_p_corr.mat'], 'taskshuffled2_gyri_p_corr');
saveas(gcf, [save_path 'taskshuffled2_gyri_p_corr_adjusted_color.png']);

group1_gs_p_corr = corrcoef(group1_gs)
% heatmap
imagesc(group1_gs_p_corr, [-0.35 1]);
colorbar;
colormap parula;
save([save_path 'group1_gs_p_corr.mat'], 'group1_gs_p_corr');
saveas(gcf, [save_path 'group1_gs_p_corr_thre_adjusted_color.png']);

group2_gs_p_corr = corrcoef(group2_gs)
% heatmap
imagesc(group2_gs_p_corr, [-0.35 1]);
colorbar;
colormap parula;
save([save_path 'group2_gs_p_corr.mat'], 'group2_gs_p_corr');
saveas(gcf, [save_path 'group2_gs_p_corr_thre_adjusted_color.png']);

taskshuffled1_gs_p_corr = corrcoef(taskshuffled1_gs)
% heatmap
imagesc(taskshuffled1_gs_p_corr, [-0.35 1]);
colorbar;
colormap parula;
save([save_path 'taskshuffled1_gs_p_corr.mat'], 'taskshuffled1_gs_p_corr');
saveas(gcf, [save_path 'taskshuffled1_gs_p_corr_adjusted_color.png']);
% saveas(gcf, [save_path 'taskshuffled1_gs_p_corr.fig']);

taskshuffled2_gs_p_corr = corrcoef(taskshuffled2_gs)
% heatmap
imagesc(taskshuffled2_gs_p_corr, [-0.35 1]);
colorbar;
colormap parula;
save([save_path 'taskshuffled2_gs_p_corr.mat'], 'taskshuffled2_gs_p_corr');
saveas(gcf, [save_path 'taskshuffled2_gs_p_corr_adjusted_color.png']);
% saveas(gcf, [save_path 'taskshuffled2_gs_p_corr.fig']);