clear all
clc
close all
addpath /data/hzb1/Projects/S900_RSN/Analyses/fft_analysis

%%
result_root_path = '/data/hzb1/Projects/S900_RSN/CNN_training/output/2-fold_CrossVal_group2/';
save_orig_path = '/data/hzb1/Projects/S900_RSN/fig/fft_view/2-fold_CrossVal_group2/';
%root_path = '/disk1/wqy/results/CNN-classifier-huan/Grayordinate/';

iStart = 1;
iEnd = 10;%68;

% before this please run get_filter.py

%%
d = zeros(1,10);
d_high = zeros(1,10);
d_lo = zeros(1,10);
for rsnid = iStart : iEnd
    if rsnid == 5 
        continue
    end
    
    disp(rsnid)
    result_path = [result_root_path, num2str(rsnid, '%02d'),'/'];
    save_path = [save_orig_path, num2str(rsnid, '%02d'),'/'];
    if ~exist(save_path,'dir')
        mkdir(save_path);
    end
    
    cd(save_path);
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
    p_gyri = P1_sum/pcount;
    
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
    p_suli = P1_sum/pcount;
    
    %% full-band Euclidean distance calculate
    o_matrix = p_gyri-p_suli;
    d2 = 0;
    for ii = 1:length(o_matrix)
        d2 = d2 + o_matrix(ii,1)^2;
    end
    d(1,rsnid) = sqrt(d2);
    
    %% high frequency Euclidean distance calculate    
    highfreqid = find(f>0.5);
    d2 = 0;
    for ii = 1:length(highfreqid)
        idx = highfreqid(ii,1);
        d2 = d2 + o_matrix(idx,1)^2;
    end
    d_high(1,rsnid) = sqrt(d2);
    
    %% low frequency Euclidean distance calculate
    lofreqid = find(f<0.1);    
    d2 = 0;
    for ii = 1:length(lofreqid)
        idx = lofreqid(ii,1);
        d2 = d2 + o_matrix(idx,1)^2;
    end
    d_lo(1,rsnid) = sqrt(d2);
end
fname = [save_orig_path,'fft_edistance.mat'];
save(fname,'d', 'd_high', 'd_lo');
