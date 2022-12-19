clear;
clc;
close all;

addpath /data1/wqy/Projects/S900_RSN/Anaylses/fft_analysis

result_root_path = '/data1/wqy/Projects/S900_RSN/CNN_training/output/contacted_training_taskshuffled/';
% save_orig_path = '/data1/wqy/Projects/S900_RSN/Results/fft/task_shuffled/';

%% compute fft
ratiomatrix = zeros(2,10);
for rsnid = 1 : 10
    if rsnid == 5 
        continue
    end
    
    disp(rsnid)
    result_path = [result_root_path, num2str(rsnid, '%02d'),'/'];
%     save_path = [save_orig_path, num2str(rsnid, '%02d'),'/'];
%     if ~exist(save_path,'dir')
%         mkdir(save_path);
%     end
%     
%     cd(save_path);
    %result_path = [result_root_path, task,'\G1\'];
    %% Conv_pred
    conv_pred = load([result_path, 'conv_pred.mat']);
    x = conv_pred.conv_pred;
    x_gyri = x(1,:,:);
    x_gyri = squeeze(x_gyri)';
    x_sulci = x(2,:,:);
    x_sulci = squeeze(x_sulci)';
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
    amp = P1_sum/pcount;
    lofreqid = find(f<0.1);
    highfreqid = find(f>0.5);
    lofreqmax = max(amp(lofreqid));
    highfreqmax = max(amp(highfreqid));
    lohigh_ratio = lofreqmax/highfreqmax;
    ratiomatrix(1,rsnid) = lohigh_ratio;
    
    % sulci
    f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    for i=1:64
        tmp = x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        avg_P1 = mean(P1);
        if avg_P1 < 1e-4
            continue
        end
        pcount = pcount + 1;
        P1_sum = P1_sum + P1;
    end
    amp = P1_sum/pcount;
    lofreqid = find(f<0.1);
    highfreqid = find(f>0.5);
    lofreqmax = max(amp(lofreqid));
    highfreqmax = max(amp(highfreqid));
    lohigh_ratio = lofreqmax/highfreqmax;
    ratiomatrix(2,rsnid) = lohigh_ratio;

end

%% anova
X = [22.4118,18.6513,19.4909,22.1769,8.5828,18.9257,16.6968,15.0043,21.6078;
    20.3207,15.5849,15.5196,19.9285,2.4817,20.8012,16.8868,14.3263,19.7389;
    13.7661,14.9100,6.9283,4.4841,2.1144,2.6810,2.7593,2.4031,9.4133;
    6.2567,15.3057,5.2519,3.3571,1.4325,2.6046,2.1364,2.8384,6.7599];%crossval1&2

% X = [22.4118,18.6513,19.4909,22.1769,8.5828,18.9257,16.6968,15.0043,21.6078;
%     20.3207,15.5849,15.5196,19.9285,2.4817,20.8012,16.8868,14.3263,19.7389;
%     19.8030,22.9315,21.2936,2.4410,18.2089,21.8686,19.5677,17.8113;
%     5.0714,19.5580,14.8083,1.9906,1.5244,2.1584,4.5568,3.0331,13.5676;
%     13.7661,14.9100,6.9283,4.4841,2.1144,2.6810,2.7593,2.4031,9.4133;
%     6.2567,15.3057,5.2519,3.3571,1.4325,2.6046,2.1364,2.8384,6.7599]%crossval1&2&task_shuffle
[p,table,stats] = anova2(X,2)