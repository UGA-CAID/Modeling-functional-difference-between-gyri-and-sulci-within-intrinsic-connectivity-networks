clear;
clc;
addpath(genpath('/data/hzb1/matlab_toolbox/vtk/'))

% for contact data
load('/data/hzb1/Projects/S900_RSN/sub_list.mat')
sub_list = sub_list';
sub_num = size(sub_list, 2);

feat_pos_portation = 0.2;
feat_neg_portation = 0.2;
feat_path = '/data/hzb1/DATA/Grayordinate_S900/Surface/'; %% feature path
rsnmask_path = '/data/hzb1/Projects/'
data_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Data_PRE/Orig_Singals_RSN_Div_Gyri_Sulci/';
surf_dir = '/data/hzb1/DATA/Grayordinate_S900/Surface/';

%%% random pick 60% subject for trainning
perc = 0.6;
ordering = randperm(sub_num);
sub_group = sub_list(ordering(1:round(sub_num*perc))); % modify here

rsnmask_fname = [rsnmask_path,'RSN.mat'] ;
   
    
for subj = 1: sub_num
%     for subj = [118,150,221,296,387]
    sub = sub_list(subj);
    feat_fname = [feat_path, num2str(sub), '/', num2str(sub),'.sulc.mat'];        
    
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
        feat_visual = feat;
        for ii =1:64984
            if RSN_mask(ii) == 0
                feat_visual(ii) = -2;
            end
        end
         %% visualization
        %read surf
        surf_L = vtkSurfRead([surf_dir, num2str(sub), '/', num2str(sub), '.L.white_MSMAll.vtk']);
        surf_R = vtkSurfRead([surf_dir, num2str(sub), '/', num2str(sub), '.R.white_MSMAll.vtk']);
        
        out_dir = ['/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/Visualization/RSN_', num2str(i,'%02d'),'/'];
        if ~exist(out_dir,'dir')
            mkdir(out_dir);
        end
        %write surf
        fname = [out_dir, num2str(sub), '.L.sulc_RSN_', num2str(i,'%02d'), '.vtk'];
        surf_L.Pdata{1,1}.val = feat_visual(1:32492,:);
        surf_L.Pdata{1,1}.name = 'GS_lable';
        surf_L.Face = surf_L.Face-1;
        vtkSurfWrite(fname, surf_L);
        
        fname = [out_dir, num2str(sub), '.R.sulc_RSN_', num2str(i,'%02d'), '.vtk'];
        surf_R.Pdata{1,1}.val = feat_visual(32493:64984,:);
        surf_R.Pdata{1,1}.name = 'GS_lable';
        surf_R.Face = surf_R.Face-1;
        vtkSurfWrite(fname, surf_R);
        
        %%
        fmri_id = find(RSN_mask>0);  
        feat_RSN = feat(fmri_id,:);
        tmp = sort(feat_RSN);
        neg_thr = tmp(floor(length(tmp)*feat_neg_portation));
        pos_thr = tmp(floor(length(tmp)*(1-feat_pos_portation)));
        
        if pos_thr<0
            display = ['sub(',num2str(sub), ') RSN_',num2str(i),' corrected.'];
            disp(display)
            pos_thr = 0;
        end
        gyri_mask = double(feat_RSN>pos_thr);
        sulci_mask = double(feat_RSN<neg_thr)*(-1);
        gs_mask = gyri_mask + sulci_mask;           
            
        visualize_matrix = zeros(64984,1);
        for tt = 1:64984
            for jj = 1:length(fmri_id)
                if tt == fmri_id(jj)
                    visualize_matrix(tt) = gs_mask(jj);
                end
            end
        end
        
        %% visualization
        %read surf
        surf_L = vtkSurfRead([surf_dir, num2str(sub), '/', num2str(sub), '.L.white_MSMAll.vtk']);
        surf_R = vtkSurfRead([surf_dir, num2str(sub), '/', num2str(sub), '.R.white_MSMAll.vtk']);
        
        out_dir = ['/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/Visualization/RSN_', num2str(i,'%02d'),'/'];
        if ~exist(out_dir,'dir')
            mkdir(out_dir);
        end
        %write surf
        fname = [out_dir, num2str(sub), '.L.GS_RSN_', num2str(i,'%02d'), '.vtk'];
        surf_L.Pdata{1,1}.val = visualize_matrix(1:32492,:);
        surf_L.Pdata{1,1}.name = 'GS_lable';
        surf_L.Face = surf_L.Face-1;
        vtkSurfWrite(fname, surf_L);
        
        fname = [out_dir, num2str(sub), '.R.GS_RSN_', num2str(i,'%02d'), '.vtk'];
        surf_R.Pdata{1,1}.val = visualize_matrix(32493:64984,:);
        surf_R.Pdata{1,1}.name = 'GS_lable';
        surf_R.Face = surf_R.Face-1;
        vtkSurfWrite(fname, surf_R);
    end    
end
