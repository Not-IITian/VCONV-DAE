clear all ;
% take the first and last viewpoint descriptor of an object
% and interpolat the intermediate ones.
no_files_per_class = 12*80 ;
% load some model of chair and first-last view point
classes = {'sofa'} ;
 data_path = 'view-interp/';

 interpolation  = 0; 
 
 if 0
     % this is for desc reading
    for i = 1: numel(classes)
    
        f_path = [data_path,classes{i}, '/encoded_desc/' ] ;
        files = dir(fullfile(f_path,'*.asc') );    
    
        for j = 1 : length(files)        
            fid = fopen(files(j).name,'rt');
        
            delimiterIn = ' ';
            headerlinesIn = 17;
%  desc_6912 = importdata(files(j).name,delimiterIn,headerlinesIn);
            desc_6912 = textscan(fid, '%f', 'HeaderLines',17);
            fclose(fid);
            desc_6912 = desc_6912{1} ;      
%         recons_sample= reshape(recons_sample, [30,30,30]) ;      
            f_name = files(j).name(1:end-4)
            save_path = [f_path,f_name, '_desc.mat' ] ;
            save(save_path, 'desc_6912')      
        end       
%         p = patch(isosurface(squeeze(out_tr),.5));
%         view(3);
%        p = patch(isosurface(squeeze(out_tr),1));
      
        %set(p,'FaceColor','red','EdgeColor','none');
       % daspect([1,1,1])
         
%         close(gcf);
    end
    
 else
     % do the interpolation 
     if interpolation         
        f_path = [data_path, classes{1},'/encoded_desc/' ] ;
        files = dir(fullfile(f_path,'*.mat') );      
        desc_mat = zeros(length(files),6912);
        
     for j = 1 : length(files) 
         load(files(j).name)
         desc_mat(j,:) = desc_6912 ;        
     end   
     
      x = [1,10] ;
       xq= [2,3,4,5,6,7,8,9] ; % replace by setdif      
     interp_desc = zeros(numel(xq),6912) ;    
       
     for i = 1:6912                
%          xq = [2] ;
%          v = [desc_mat(1,i), desc_mat(2,i) ];
%             vq = interp1(x,v,xq,'linear') ;
%          interp_desc(i) = vq ;
        diff = desc_mat(1,i) - desc_mat(2,i);
        diff_increment = diff./(x(end)-x(1)) ;
        vec_diff =  cumsum(diff_increment*ones(numel(xq),1)) ;
        interp_desc(:,i) =  desc_mat(2,i) + vec_diff ;        
     end 
     
     save_desc = ['view-interp/', classes{1}, '/encoded_desc/', 'interpolated_desc.mat'];
     save(save_desc, 'interp_desc') ;
    
     else
         % do the visualization         
        f_path = [data_path, classes{1},'/interp_desc/'] ;
        files = dir(fullfile(f_path,'*.asc') );    
    
        for j = 1 : length(files) 
            fid = fopen(files(j).name,'rt');       
            delimiterIn = ' ';  
            headerlinesIn = 17;
            recons_sample = textscan(fid, '%f', 'HeaderLines',17);
            fclose(fid);
            recons_sample = recons_sample{1} ;      
            recons_sample= reshape(recons_sample, [30,30,30]) ;      
            f_name = files(j).name(1:end-4)               
            figure ;      
            p = isosurface(squeeze(recons_sample),0.5) ;
            patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
  
            axis equal 
            axis off
            lighting gouraud
        end       
%        

%         load('bed_0019_1.mat') ;        
%         
%         figure ;      
%             p = isosurface(squeeze(instance),0.5) ;
%             patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
%   
%             axis equal 
%             axis off
%             lighting gouraud
       
    end
                     
   end    
