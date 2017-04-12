% this scripts is used to visualize the shape completion output figures as shown in the paper.
classes = {  'desk', 'bathtub', 'toilet', 'monitor', 'night_stand', 'table', 'sofa', 'dresser', 'bed', 'chair'} ; 
classes_len = [4, 7, 6,  7, 11, 5, 4, 7, 3, 5] ; %   used later
% classes = {'chair'} ;

 data_path = 'recons/';  
 noise_level = '50';  % needed for file saving later
 noise_type = 'rand' ;
if 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS SECTION READS RECONSTRUCTED BINARY FILE AND CONVERTS IT TO MAT FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
for i = 1: numel(classes)  
    d_path = [data_path,'mat-files-paper-',noise_type,'/'];
    f_path = [d_path,classes{i}, '/' ] ;
    files = dir( fullfile(f_path,'*.asc') );  
    
    for j = 1 : length(files)       
        fid = fopen(files(j).name,'rt');
        recons_sample = textscan(fid, '%f', 'HeaderLines',17);
         fclose(fid);
        recons_sample = recons_sample{1} ;
        recons_sample= reshape(recons_sample, [30,30,30]) ;
        
        f_name = files(j).name(1:end-4)
        save_path = [f_path,f_name, '.mat' ] ;
        save(save_path, 'recons_sample')
        
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%THIS PART IS FOR VISUALISATION %%%%%%%%%%%%%%%%%%%%%%%%%%

 
for i = 1:numel(classes)
    
%     f_path = [data_path,'mat-files-paper-',noise_type,'/',classes{i}, noise_level, '/' ] ;
        f_path = [data_path,'mat-files-paper-',noise_type,'/',classes{i}, '/' ] ;
    
        files = dir( fullfile(f_path,'*.mat') );    
        data_file = ['Data/',classes{i}, '_te.mat' ] ;
        load(data_file)
        
        dist_file = ['Data/', 'rand_', classes{i}, '_te.mat' ];    
        load(dist_file)
        
        for j = 1 : length(files) 
        
        init_length = classes_len(i) + 10;
        files(j).name
        idx = files(j).name(init_length:end-4)      
        idx = str2num(idx) ;
        
        the_sample = te_data(idx,:,:,:);   
        dist_sample = te_distorted_50(idx,:,:,:) ;
         load (files(j).name)
           
        figure;    
        p = isosurface(squeeze(the_sample),0.5) ;
         patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3)  
         
        axis equal 
        axis off      
        lighting gouraud   
        figure ;      
        
         p = isosurface(squeeze(dist_sample),0.5) ;
        patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
  
        axis equal 
        axis off       
        lighting gouraud  
        figure;  
        
        p = isosurface(squeeze(recons_sample),0.5) ;
        patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
  
          axis equal
          axis off
        lighting gouraud

        pause;

    close all;
       
        end
end


% classes = {  'desk', 'bathtub', 'toilet', 'monitor', 'night_stand', 'table', 'sofa', 'dresser', 'bed', 'chair'} ;
% Differences between original and reconstructed
%   VD = abs(J-I);
%
%   % Show the original Mesh and Mesh of new volume
%   figure, 
%   subplot(1,3,1),  title('original')
%     patch(FV,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
%   subplot(1,3,2), title('converted');
%     patch(isosurface(J,0.8),'facecolor',[0 0 1],'edgecolor','none'), camlight;view(3);
%   subplot(1,3,3), title('difference');
%     patch(isosurface(VD,0.8),'facecolor',[0 0 1],'edgecolor','none'), camlight;view(3); 
