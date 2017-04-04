load 'monitor_tr.mat'

dims = size(tr_data);
no_samples = dims(1) ;
%no_samples = 20;
avg_occupancy = 0;
occupancy_later = 0;
tr_distorted_30 = zeros(dims);

structured_dataset = 0;
% find the min max of extent of each example along each direction

for n=10:10

  the_sample = tr_data(n,:,:,:);
  the_sample = squeeze(the_sample) ;
  
   slice_previous = [] ;
  % first go thourgh each dims
  
    for i = 1:30
        
        slice = the_sample(i,:,:) ;
        slice = squeeze(slice) ;  %slice is a matrix now
             
       % assuming when 0 to 1 happens, shape enter while 1 to 0 happens
       % only when shape exits
        if numel(find(slice)) >0 && numel(find(slice_previous))== 0
            min_x = i ;           
        elseif  numel(find(slice)) ==0 && numel(find(slice_previous))> 0         
            max_x = i ;          
        end 
        
    slice_previous = slice ;
        
    end    
    
    slice_previous = [] ;
    
    for j = 1:30
        slice = the_sample(:,j,:) ;
        slice = squeeze(slice) ;  %slice is a matrix now
             
       % assuming when 0 to 1 happens, shape enter while 1 to 0 happens
       % only when shape exits
        if numel(find(slice)) >0 && numel(find(slice_previous))== 0
            min_y = j ;        
            
        elseif  numel(find(slice)) ==0 && numel(find(slice_previous))> 0  
            
            max_y = j ;
            
        end      
    slice_previous = slice ;
        
    end 
    
    slice_previous = [] ;
    
    for k = 1:30
        slice = the_sample(:,:,k) ;
        slice = squeeze(slice) ;  %slice is a matrix now
        
       % assuming when 0 to 1 happens, shape enter while 1 to 0 happens
       % only when shape exits
       
        if numel(find(slice)) >0 && numel(find(slice_previous))== 0
            min_z = k ;        
        elseif  numel(find(slice)) ==0 && numel(find(slice_previous))> 0    
            max_z = k ;
            
        end   
        
    slice_previous = slice ;      
    end 
         
end

if 1
% now set some fraction to zero
% for i=1:no_sample   
     the_sample = tr_data(100,:,:,:);
     
     filled_voxels = numel(find(the_sample)) ;
     avg_occupancy = avg_occupancy + filled_voxels ;
     aa = the_sample;   
       
     dim_extent = max_y -min_y ; 
     no_slices = ceil(.3*dim_extent) ;     
     
     slice_idx = [min_y];
     
     for jj = 1:no_slices
         slice = jj + min_y ;
         slice_idx = [slice_idx, slice] ;                
     end  
     
     the_sample(:,:,slice_idx,: ) = 0;
     occ_voxels = numel(find(the_sample)) ;
     occupancy_later =  occ_voxels + occupancy_later ;
        
%      display 'next iteration'
%      tr_distorted_30(i,:,:,:) = the_sample ;          
%      
% end

figure;
%          subplot(1,2,2);
        plot3D(squeeze(the_sample) > 0);
        daspect([1,1,1])
        view(3); 

% avg_occupancy = avg_occupancy/no_samples 
% occupancy_later = occupancy_later/no_samples

end