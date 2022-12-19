clear all
clc
close all
addpath /data/hzb1/Projects/S900_RSN/Analyses/fft_analysis

%%
result_root_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1/';
save_orig_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/fig/fft_view/Contact_Group_1/';
%root_path = '/disk1/wqy/results/CNN-classifier-huan/Grayordinate/';

iStart = 1;
iEnd = 10;%68;

% before this please run get_filter.py

%%
% for rsnid = 2 : 2
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
    
% %% y axis limit
%     %% sulci fft plot
%     axes1 = axes;
%     hold(axes1,'on');
%     for i=1:64
% %         subplot(8,8,i); 
%         tmp = x_sulci(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         plot(f,P1,'LineWidth', 1.5);
%         ylim([0 0.1]);
%         hold on;
%     end
%     grid on;
%     set(axes1,'FontName','Times New Roman','FontSize',22);
%     set(gcf, 'position', [0 0 1500 1000]);
%     
%     xlabel('f(Hz)','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
%     ylabel('Amplitude','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
%     if rsnid<5
%         title(['ICN ',num2str(rsnid, '%d'),' Sulci'],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     else
%         title(['ICN ',num2str(rsnid-1, '%d'),' Sulci'],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     end 
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft.png'];
%     saveas(gcf,figname);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft.fig'];
%     saveas(gcf,figname);
%     clf;
%     
%     %% gyri fft plot
%     axes1 = axes;
%     hold(axes1,'on');
%     for i=1:64
% %         subplot(8,8,i); 
%         tmp = x_gyri(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         plot(f,P1,'LineWidth', 1.5);
%         ylim([0 0.1]);
%         hold on;
%     end
%     grid on;
%     set(axes1,'FontName','Times New Roman','FontSize',22);
%     set(gcf, 'position', [0 0 1500 1000]);
%     
%     xlabel('f(Hz)','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
%     ylabel('Amplitude','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
%     if rsnid<5
%         title(['ICN ',num2str(rsnid, '%d'),' Gyri'],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     else
%         title(['ICN ',num2str(rsnid-1, '%d'),' Gyri'],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     end    
%     
%     figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft.png'];
%     saveas(gcf,figname);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft.fig'];
%     saveas(gcf,figname);
%     clf;

%% y axis limit only curves
    %% sulci fft plot
    axes1 = axes;
    hold(axes1,'on');
    for i=1:64
%         subplot(8,8,i); 
        tmp = x_sulci(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        plot(f,P1,'LineWidth', 5);
        ylim([0 0.1]);
        hold on;
    end
    set(gcf, 'position', [0 0 1500 1000]);   
    axes1.YAxis.Visible = 'off';
    axes1.XAxis.Visible = 'off';
    
    figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft_only_curves.png'];
    saveas(gcf,figname);
    figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft_only_curves.fig'];
    saveas(gcf,figname);
    clf;
    
    %% gyri fft plot
    axes1 = axes;
    hold(axes1,'on');
    for i=1:64
%         subplot(8,8,i); 
        tmp = x_gyri(:,i);
        [f, P1, ~] = myfft(tmp, 0.72,i);
        plot(f,P1,'LineWidth', 5);
        ylim([0 0.1]);
        hold on;
    end
    set(gcf, 'position', [0 0 1500 1000]);   
    axes1.YAxis.Visible = 'off';
    axes1.XAxis.Visible = 'off';
   
    figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft_only_curves.png'];
    saveas(gcf,figname);
    figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft_only_curves.fig'];
    saveas(gcf,figname);
    clf;

%     %% compute average fft for filter 0 excluded
%     % gyri
%     f_sum = zeros(33,1);
%     P1_sum = zeros(33,1);
%     pcount = 0;
%     for i=1:64
%         tmp = x_gyri(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         avg_P1 = mean(P1);
%         if avg_P1 < 1e-4
%             continue
%         end
%         pcount = pcount + 1;       
%         P1_sum = P1_sum + P1;
%     end
%     plot(f, P1_sum/pcount, 'LineWidth', 2 , 'color', [0.635294139385223 0.423529416322708 0.419607847929001]);
%     xlim([0 1]);
%     ylim([0 0.05]);
%     box off;
%     set(gcf, 'position', [0 0 1500 1000]);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft_avg_0excluded.png'];
%     saveas(gcf,figname);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.gyri.fft_avg_0excluded.fig'];
%     saveas(gcf,figname);
%     
%     % sulci
%     f_sum = zeros(33,1);
%     P1_sum = zeros(33,1);
%     pcount = 0;
%     for i=1:64
%         tmp = x_sulci(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         avg_P1 = mean(P1);
%         if avg_P1 < 1e-4
%             continue
%         end
%         pcount = pcount + 1;
%         P1_sum = P1_sum + P1;
%     end
%     plot(f, P1_sum/pcount, 'LineWidth', 2 , 'color', [0.301960784313725 0.301960784313725 0.301960784313725]);
%     xlim([0 1]);
%     ylim([0 0.05]);
%     box off;
% %     set(gcf, 'position', [0 0 1500 1000]);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft_avg_0excluded.png'];
%     saveas(gcf,figname);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci.fft_avg_0excluded.fig'];
%     saveas(gcf,figname);

%     %% compute average fft for filter 0 excluded
%     % combine sulci and gyri
%     f_sum = zeros(33,1);
%     P1_sum = zeros(33,1);
%     pcount = 0;
%     axes1 = axes;
%     hold(axes1,'on');
%     for i=1:64
%         tmp = x_gyri(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         avg_P1 = mean(P1);
%         if avg_P1 < 1e-4
%             continue
%         end
%         pcount = pcount + 1;       
%         P1_sum = P1_sum + P1;
%     end
%     plot(f, P1_sum/pcount, 'LineWidth', 2 , 'color', [0.635294139385223 0.423529416322708 0.419607847929001]);
%     hold on;
%     
%     % sulci
%     f_sum = zeros(33,1);
%     P1_sum = zeros(33,1);
%     pcount = 0;
%     for i=1:64
%         tmp = x_sulci(:,i);
%         [f, P1, ~] = myfft(tmp, 0.72,i);
%         avg_P1 = mean(P1);
%         if avg_P1 < 1e-4
%             continue
%         end
%         pcount = pcount + 1;
%         P1_sum = P1_sum + P1;
%     end
%     
%     box off;
%     set(axes1,'FontName','Times New Roman','FontSize',22);
%     set(gcf, 'position', [0 0 1500 1000]);
%     
%     plot(f, P1_sum/pcount, 'LineWidth', 2 , 'color', [0.301960784313725 0.301960784313725 0.301960784313725]);
%     xlabel('f(Hz)','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
%     ylabel('Amplitude','Fontname','Times New Roma','FontSize',28,'FontWeight','bold');
% %     xlim([0 1]);
%     ylim([0 0.06]);
%     if rsnid<5
%         title(['ICN ',num2str(rsnid, '%d')],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     else
%         title(['ICN ',num2str(rsnid-1, '%d')],'Fontname','Times New Roma','FontSize',32,'FontWeight','bold');
%     end
%     
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci&gyri.fft_avg_0excluded.png'];
%     saveas(gcf,figname);
%     figname = [num2str(rsnid, '%d'), '.conv_pred.sulci&gyri.fft_avg_0excluded.fig'];
%     saveas(gcf,figname);
%     clf;
    
    %% compute average fft for filter 0 excluded
    % combine sulci and gyri no axes
    f_sum = zeros(33,1);
    P1_sum = zeros(33,1);
    pcount = 0;
    axes1 = axes;
    hold(axes1,'on');
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
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [0.635294139385223 0.423529416322708 0.419607847929001]);
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', 'r');
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [159/256 2/256 30/256]);
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [68/256 114/256 196/256]);
    plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', 'k');
    hold on;
    
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
    
    box off;
    axes1.YAxis.Visible = 'off';
    axes1.XAxis.Visible = 'off';
    set(gcf, 'position', [0 0 1500 1000]);
    
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [0.301960784313725 0.301960784313725 0.301960784313725]);
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', 'b');
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [51/256 64/256 158/256]);
%     plot(f, P1_sum/pcount, 'LineWidth', 8 , 'color', [237/256 125/256 49/256]);
    plot(f, P1_sum/pcount,':', 'LineWidth', 8 , 'color', 'r');
    ylim([0 0.06]);
    
    figname = [num2str(rsnid, '%d'), '.conv_pred.sulci&gyri.fft_avg_0excluded_onlycurves.png'];
    saveas(gcf,figname);
    figname = [num2str(rsnid, '%d'), '.conv_pred.sulci&gyri.fft_avg_0excluded_onlycurves.fig'];
    saveas(gcf,figname);
    clf;
end

