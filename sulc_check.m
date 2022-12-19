clear;
clc;

tasks = {'EMOTION';'GAMBLING';'LANGUAGE';'MOTOR';'RELATIONAL';'SOCIAL';'WM'};

% for contact data
load('/data/hzb1/Projects/S900_RSN/sub_list.mat')
sub_list = sub_list';
sub_num = size(sub_list, 2);

feat_pos_portation = 0.3;
feat_neg_portation = 0.3;
feat_path = '/data/hzb1/DATA/Grayordinate_S900/Surface/'; %% feature path
rsnmask_path = '/data/hzb1/Projects/'
data_path = '/data/hzb1/Projects/S900_RSN/Data_PRE/Orig_Singals_RSN_Div_Gyri_Sulci/';
save_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/sulc_check/';

%%% random pick 60% subject for trainning
perc = 0.6;
ordering = randperm(sub_num);
sub_group = sub_list(ordering(1:round(sub_num*perc))); % modify here

rsnmask_fname = [rsnmask_path,'RSN.mat'] ;
for tt = 1:1
    disp(tasks{tt})
    %% specify
    task = tasks{tt};
    fMRI_path = ['/data/hzb1/DATA/Grayordinate_S900/',task, '/'];
    
    %for subj = 1:size(sub_group,1)
        %sub = sub_group(subj);
    check_gyri_matrix = zeros(sub_num,10);
    check_sulci_matrix = zeros(sub_num,10);
    for subj = 11: 16
%     for subj = 1: sub_num
        sub = sub_list(subj);
        fMRI_fname = [fMRI_path,num2str(sub),'.mat'];
        feat_fname = [feat_path, num2str(sub), '/', num2str(sub),'.sulc.mat'];
        
        fMRI = load(fMRI_fname);
        fMRI = fMRI.data';
        feat = load(feat_fname);
        feat = feat.sulc;
                
        disp('RSN')
        for i = 6:6
%         for i = 1:10
            if i==5
                continue;
            end
            display = ['for sub(',num2str(sub), ') RSN_',num2str(i),' check sulci and gyri'];
            disp(display)
            
            RSN_mask = load(rsnmask_fname);
            RSN_mask = RSN_mask.RSN;
            RSN_mask = RSN_mask{i};
            fmri_id = find(RSN_mask>0);
            fmri_RSN = fMRI(:,find(RSN_mask>0));
            feat_RSN = feat(fmri_id,:);
            tmp = sort(feat_RSN);
            neg_thr = tmp(floor(length(tmp)*feat_neg_portation));
            pos_thr = tmp(floor(length(tmp)*(1-feat_pos_portation)));
            
            check_gyri_matrix(subj, i) = pos_thr;
            check_sulci_matrix(subj, i) = neg_thr;            
            
        end

    end
%     save([save_path,'check_matrix.mat'],'check_gyri_matrix','check_sulci_matrix');
end