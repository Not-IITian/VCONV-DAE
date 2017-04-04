load('sofa_te.mat')
%  
dims = size(te_data);
n = dims(1);

load('dist_sofa_te.mat')
     
[no_samples, ~,~,~] = size(te_data) ;


        
for i = 1:6:n
    
    the_sample = te_data(i,:,:,:);   
        dist_sample = te_distorted_30(i,:,:,:) ;
    
    figure,
    subplot(2,1,1); title('distorted');
   p = isosurface(squeeze(dist_sample),0.5) ;
  patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
  
        axis equal
        axis off
        lighting gouraud
        
   subplot(2,1,2); title('original');
   p = isosurface(squeeze(the_sample),0.5) ;
  patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
  
        axis equal
        axis off
        lighting gouraud
        i
   pause
   
   
    
end  
% subplot(2,1,2); title('reconstructed');
%    p = isosurface(squeeze(recons_sample),0.5) ;
%   patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3);
%   
%           axis tight
%         lighting gouraud
% 
%     pause;
%     endclear all
%     