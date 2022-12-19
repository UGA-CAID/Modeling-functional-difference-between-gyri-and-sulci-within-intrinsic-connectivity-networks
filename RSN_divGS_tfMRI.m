clear;
clc;

tasks = {'EMOTION';'GAMBLING';'LANGUAGE';'MOTOR';'RELATIONAL';'SOCIAL';'WM'};

% for contact data
load('/data/hzb1/Projects/S900_RSN/sub_list.mat')
sub_list = sub_list';
sub_num = size(sub_list, 2);

feat_pos_portation = 0.2;
feat_neg_portation = 0.2;
feat_path = '/data/hzb1/DATA/Grayordinate_S900/Surface/'; %% feature path
rsnmask_path = '/data/hzb1/Projects/'
data_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Data_PRE/Orig_Singals_RSN_Div_Gyri_Sulci/';

%%% random pick 60% subject for trainning
perc = 0.6;
ordering = randperm(sub_num);
sub_group = sub_list(ordering(1:round(sub_num*perc))); % modify here

rsnmask_fname = [rsnmask_path,'RSN.mat'] ;
for tt = 1:7
    disp(tasks{tt})
    %% specify
    task = tasks{tt};
    fMRI_path = ['/data/hzb1/DATA/Grayordinate_S900/',task, '/'];
    
    %for subj = 1:size(sub_group,1)
        %sub = sub_group(subj);
    for subj = 1: sub_num
%     for subj = [118,150,221,296,387]
        sub = sub_list(subj);
        fMRI_fname = [fMRI_path, num2str(sub), '.mat'];
        feat_fname = [feat_path, num2str(sub), '/', num2str(sub),'.sulc.mat'];
        
        fMRI = load(fMRI_fname);
        fMRI = fMRI.data';
        feat = load(feat_fname);
        feat = feat.sulc;
                
        disp('RSN')
        for i = 1:10
            if i==5
                continue;
            end
            display = ['for sub( ',num2str(sub), ') RSN_',num2str(i),' divide sulci and gyri'];
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
            
            if pos_thr<0
                display = ['sub(',num2str(sub), ') RSN_',num2str(i),' corrected.'];
                disp(display)
                pos_thr = 0;
%                 gyri_mask_1 = feat_RSN>pos_thr;
%                 gyri_mask_2 = 0>feat_RSN;
%                 gyri_mask = double(gyri_mask_1&gyri_mask_2); 
%                 count = sum(gyri_mask)
            end
            gyri_mask = double(feat_RSN>pos_thr);
            sulci_mask = double(feat_RSN<neg_thr);
            fMRI_gyri = zscore(fmri_RSN(:,find(gyri_mask>0)));
            fMRI_sulci = zscore(fmri_RSN(:,find(sulci_mask>0)));            
           
            gyri_nonzero = find(std(fMRI_gyri) > eps*1e5);
            sulci_nonzero = find(std(fMRI_sulci) > eps*1e5);
            fMRI_gyri = fMRI_gyri(:,gyri_nonzero);
            fMRI_sulci = fMRI_sulci(:,sulci_nonzero);
            mkdir([data_path,'RSN_',num2str(i,'%02d'),'/',task,'/']);
            save([data_path,'RSN_',num2str(i,'%02d'),'/',task,'/',num2str(sub),'.mat'],'fMRI_gyri','fMRI_sulci');
        end

    end
    
end