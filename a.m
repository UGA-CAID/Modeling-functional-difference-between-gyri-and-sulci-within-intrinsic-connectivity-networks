clear;
clc;

tasks = {'EMOTION';'GAMBLING';'LANGUAGE';'MOTOR';'RELATIONAL';'SOCIAL';'WM'};

% sub_lists={[], % EMOTION
%     [],%GAMBLING
%     [], %LANGUAGE
%     [], % MOTOR
%     [], %RELATIONAL
%     [], %SOCIAL
%     []}; %WM
% subs_num = [692, 689, 718, 761, 781, 785, 794]; % for each task.
load('/data/hzb1/Projects/S900_RSN/sub_list.mat');
sub_list = sub_list';
sub_num = size(sub_list, 2);

data_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Data_PRE/Orig_Singals_RSN_Div_Gyri_Sulci/';
save_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Train_MatData/Group_for_Contact_Train_adjusted/';
train_perc = 0.5;
train_group = load('/data/hzb1/Projects/S900_RSN/Train_MatData/Group_for_Contact_Train/train_perc_0.5/train_group.mat');
train_group = train_group.train_group;
test_group = load('/data/hzb1/Projects/S900_RSN/Train_MatData/Group_for_Contact_Train/train_perc_0.5/test_group.mat');
test_group = test_group.test_group;
% ordering = randperm(sub_num);
% train_group = sub_list(ordering(1:round(sub_num*train_perc))); % modify here
% test_group = sub_list(ordering((round(sub_num*train_perc)+1):sub_num));

for tt = 1:7
    disp(tasks{tt})
    %% specify
    task = tasks{tt};
    %mkdir([save_path,task]);
    out_path = [save_path,'train_perc_', num2str(train_perc),'/',task];
    if ~exist(out_path,'dir')
        mkdir(out_path);
    end    
    
    for rsn = 1:10
        if rsn == 5
            continue
        end
        disp('.')
        %%train_group
        fMRI_gyri_train = [];
        fMRI_sulci_train = [];
        for subj = 1:length(train_group)
            sub = train_group(subj);
            signal_fname = [data_path,'RSN_',num2str(rsn,'%02d'),'/',task,'/',num2str(sub),'.mat'];
            load(signal_fname)
            
            min_num = min(size(fMRI_gyri,2),size(fMRI_sulci,2));
            gyri_shuffle = randperm(size(fMRI_gyri,2));
            sulci_shuffle = randperm(size(fMRI_sulci,2));
            fMRI_gyri = fMRI_gyri(:, gyri_shuffle);
            fMRI_sulci = fMRI_sulci(:,sulci_shuffle);
            
            fMRI_gyri_train = [fMRI_gyri_train,fMRI_gyri(:,1:min_num)];
            fMRI_sulci_train = [fMRI_sulci_train,fMRI_sulci(:,1:min_num)];
        end
        
%         min_num = min(size(fMRI_gyri_train,2),size(fMRI_sulci_train,2));
%         gyri_shuffle = randperm(size(fMRI_train_gyri_tmp,2));
%         sulci_shuffle = randperm(size(fMRI_train_sulci_tmp,2));
%         fMRI_gyri_train = fMRI_train_gyri_tmp(:,gyri_shuffle);
%         fMRI_sulci_train = fMRI_train_sulci_tmp(:,sulci_shuffle);
        
        fMRI_train = [fMRI_gyri_train,fMRI_sulci_train];
        label_train = [zeros(1,size(fMRI_gyri_train,2)),ones(1,size(fMRI_sulci_train,2))];
        fMRI_train = fMRI_train';
        label_train = label_train';
%         train_shuffle = randperm(min_num*2);
%         fMRI_train = fMRI_train(:,train_shuffle)';
%         label_train = label_train(:,train_shuffle)';
        
        save([out_path,'/RSN_',num2str(rsn,'%02d'),'_train_perc_',num2str(train_perc),'.mat'],'fMRI_train','label_train','-v7.3')
        
        %%test_group
        fMRI_gyri_test = [];
        fMRI_sulci_test = [];
        for subj = 1:length(test_group)
            sub = test_group(subj);       
            signal_fname = [data_path,'RSN_',num2str(rsn,'%02d'),'/',task,'/',num2str(sub),'.mat'];
            load(signal_fname)
            
            min_num = min(size(fMRI_gyri,2),size(fMRI_sulci,2));
            gyri_shuffle = randperm(size(fMRI_gyri,2));
            sulci_shuffle = randperm(size(fMRI_sulci,2));
            fMRI_gyri = fMRI_gyri(:, gyri_shuffle);
            fMRI_sulci = fMRI_sulci(:,sulci_shuffle);
            
            fMRI_gyri_test = [fMRI_gyri_test,fMRI_gyri(:,1:min_num)];
            fMRI_sulci_test = [fMRI_sulci_test,fMRI_sulci(:,1:min_num)];
            
        end
        
%         min_num = min(size(fMRI_gyri_test,2),size(fMRI_sulci_test,2));
%         gyri_shuffle = randperm(size(fMRI_test_gyri_tmp,2));
%         sulci_shuffle = randperm(size(fMRI_test_sulci_tmp,2));
%         fMRI_gyri_test = fMRI_test_gyri_tmp(:,gyri_shuffle);
%         fMRI_sulci_test = fMRI_test_sulci_tmp(:,sulci_shuffle);
        
        fMRI_test = [fMRI_gyri_test,fMRI_sulci_test];
        label_test = [zeros(1,size(fMRI_gyri_test,2)),ones(1,size(fMRI_sulci_test,2))];
        fMRI_test = fMRI_test';
        label_test = label_test';
%         test_shuffle = randperm(min_num*2);
%         fMRI_test = fMRI_test(:,test_shuffle)';
%         label_test = label_test(:,test_shuffle)';
        
        save([out_path,'/RSN_',num2str(rsn,'%02d'),'_test_perc_',num2str(1-train_perc),'.mat'],'fMRI_test','label_test','-v7.3')
    end
    
end