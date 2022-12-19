clear;
clc;

tasks = {'EMOTION';'GAMBLING';'LANGUAGE';'MOTOR';'RELATIONAL';'SOCIAL';'WM'};

% for contact data
load('/data/hzb1/Projects/S900_RSN/sub_list.mat')
sub_list = sub_list';
sub_num = size(sub_list, 2);

%%
data_path = '/data/hzb1/Projects/S900_RSN/Data_PRE/Orig_Singals_RSN_Div_Gyri_Sulci/';
save_path = '/data/hzb1/Projects/S900_RSN/Train_MatData/Group_for_Contact_Train/';%modify
train_perc = 0.8;

ordering = randperm(sub_num);
train_group = sub_list(ordering(1:round(sub_num*train_perc))); % modify here
test_group = sub_list(ordering((round(sub_num*train_perc)+1):sub_num));

for tt = 1:7
    fprintf(['\n',tasks{tt}])
%     disp(tasks{tt})
    %% specify
    task = tasks{tt};
    %sub_list = sub_lists{tt}; %comment because of contacting data prerpare
    %sub_num = subs_num(tt);%comment because of contacting data prerpare
    %mkdir([save_path,task]);
    out_path = [save_path,'/train_perc_', num2str(train_perc),'/',task];
    mkdir(out_path);

    for rsn = 1:10
        if rsn ==5
            continue
        end
        fprintf('.')
        %%train_group
        fMRI_gyri_train = [];
        fMRI_sulci_train = [];
        for sub = train_group
            signal_fname = [data_path,'RSN_',num2str(rsn,'%02d'),'/',task,'/',num2str(sub),'.mat'];
            load(signal_fname)
            fMRI_gyri_train = [fMRI_gyri_train,fMRI_gyri];
            fMRI_sulci_train = [fMRI_sulci_train,fMRI_sulci];
        end
        
        min_num = min(size(fMRI_gyri_train,2),size(fMRI_sulci_train,2));
%         gyri_shuffle = randperm(size(fMRI_train_gyri_tmp,2));
%         sulci_shuffle = randperm(size(fMRI_train_sulci_tmp,2));
%         fMRI_gyri_train = fMRI_train_gyri_tmp(:,gyri_shuffle);
%         fMRI_sulci_train = fMRI_train_sulci_tmp(:,sulci_shuffle);
        
        fMRI_train = [fMRI_gyri_train(:,1:min_num),fMRI_sulci_train(:,1:min_num)];
        label_train = [zeros(1,min_num),ones(1,min_num)];
        fMRI_train = fMRI_train';
        label_train = label_train';
%         train_shuffle = randperm(min_num*2);
%         fMRI_train = fMRI_train(:,train_shuffle)';
%         label_train = label_train(:,train_shuffle)';
        
         save([out_path,'/RSN_',num2str(rsn,'%02d'),'_train_perc_',num2str(train_perc),'.mat'],'fMRI_train','label_train', '-v7.3')
        
        %%test_group
        fMRI_gyri_test = [];
        fMRI_sulci_test = [];
        for sub = test_group
            signal_fname = [data_path,'RSN_',num2str(rsn,'%02d'),'/',task,'/',num2str(sub),'.mat'];
            load(signal_fname)
            fMRI_gyri_test = [fMRI_gyri_test,fMRI_gyri];
            fMRI_sulci_test = [fMRI_sulci_test,fMRI_sulci];
            
        end
        
        min_num = min(size(fMRI_gyri_test,2),size(fMRI_sulci_test,2));
%         gyri_shuffle = randperm(size(fMRI_test_gyri_tmp,2));
%         sulci_shuffle = randperm(size(fMRI_test_sulci_tmp,2));
%         fMRI_gyri_test = fMRI_test_gyri_tmp(:,gyri_shuffle);
%         fMRI_sulci_test = fMRI_test_sulci_tmp(:,sulci_shuffle);
        
        fMRI_test = [fMRI_gyri_test(:,1:min_num),fMRI_sulci_test(:,1:min_num)];
        label_test = [zeros(1,min_num),ones(1,min_num)];
        fMRI_test = fMRI_test';
        label_test = label_test';
%         test_shuffle = randperm(min_num*2);
%         fMRI_test = fMRI_test(:,test_shuffle)';
%         label_test = label_test(:,test_shuffle)';
        
         save([out_path,'/RSN_',num2str(rsn,'%02d'),'_test_perc_',num2str(1-train_perc),'.mat'],'fMRI_test','label_test', '-v7.3')
    end
    
end                        
