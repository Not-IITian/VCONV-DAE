
% boolean flag description
structured_dataset = 0;
multi_class_data = 0 ;
data_path = '/BS/deep_3d/work/deep_3d/fcn_3D/Data/';
classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};

slicing_noise_data = 0 ;
% classnames = { 'bathtub',  'chair', 'desk', 'dresser', 'night_stand', 'toilet'};
noise_levels = {'10', '20' , '30' } ;
zero_one_noise = 0;
% param finish

files = dir( fullfile(data_path,'*.mat') );
for j = 1 : length(files)              
              % you only want to select the files that start with class
              % name and end with _te                          
                if strcmp(files(j).name(1:5), 'rand_') ||strcmp(files(j).name(1:5), 'dist_') ||~strcmp(files(j).name(end-6:end), '_te.mat')                    
                        continue;
                    end
                                  
%                  
                 str_len = length(files(j).name) ;
                    class_name = files(j).name(1:str_len-7) 
                    assert(any(ismember(classnames,class_name ))) ;        
                load(files(j).name)   % load the original file
                dims = size(te_data);
                no_samples = dims(1) ; 
                te_distorted_10 = zeros(dims);
                te_distorted_20 = zeros(dims);
                te_distorted_30 = zeros(dims);
                
                if slicing_noise_data
                    noisy_file_name = ['dist_', class_name,'_te.mat']                              
                    %no_samples = 20;
                    avg_occupancy = 0;
                    occupancy_later = 0;
                                        
                    for i=1:no_samples
                        the_sample = te_data(i,:,:,:);     
%                        filled_voxels = numel(find(the_sample)) 
%                         avg_occupancy = avg_occupancy + filled_voxels ;
                        aa = the_sample;   
                             if structured_dataset == 0 
                             % this is for random slicing noise                            
                              n = randperm(3,1) ; % this is to choose a random direction
                                % now decide
                                 bb = randperm(30,3);
                                  cc = randperm(30,6);
                                   dd = randperm(30,9);
                                
                                switch n
                                    case 1  % slicing along x axis
                                       the_sample(:,bb,:, :) = 0;
                                       te_distorted_10(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,cc,:, :) = 0;
                                       te_distorted_20(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,dd,:, :) = 0;
                                       te_distorted_30(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                           
                                    case 2 % slicing along y axis
                                        the_sample(:,:,bb, :) = 0;
                                       te_distorted_10(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,:,cc, :) = 0;
                                       te_distorted_20(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,:,dd, :) = 0;
                                       te_distorted_30(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                    case 3 % slicing along z axis
                                        
                                        the_sample(:,:,:, bb) = 0;
                                       te_distorted_10(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,:,:, cc) = 0;
                                       te_distorted_20(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                        
                                        the_sample(:,:,:, dd) = 0;
                                       te_distorted_30(i,:,:,:) = the_sample ;
                                        the_sample = te_data(i,:,:,:);
                                                                                                               
                                end
                              
                                
%                                 occ_voxels = numel(find(the_sample)) ;
%                                 occupancy_later =  occ_voxels + occupancy_later ;
%         
%                                 display 'next iteration'                        
                         end                       
                    end
                    
                else
                % this is for random noise
                % you can insert it two types: 1. shutting down voxels (1-0 noise)
                % 2. shutting down as well as opening up noisy voxels (0-1 noise)
                  te_distorted_50 = zeros(dims);  
                noisy_file_name = [data_path ,'rand_20_', class_name,'_te.mat']     ;
                 for n=1:no_samples
                        the_sample = te_data(n,:,:,:);     
%                         filled_voxels = numel(find(the_sample)) 
%                         avg_occupancy = avg_occupancy + filled_voxels ;
%                         aa = the_sample;                        
                            noisy_idx_50 =  randperm(27000,13500) ;
                             noisy_idx_20 = randperm(27000, 5400) ;
                             noisy_idx_30 = randperm(27000,8100) ;
                            
                        if zero_one_noise
                            % flip both values 0 to 1 and 1 to 0
                            
                            
                            
                        else
                            % flip 1 to 0 ;
                            dummy = the_sample ;
                            dummy(noisy_idx_50) = 0 ;
                            te_distorted_50(n,:,:,:) = dummy ;  
                            
                            dummy = the_sample ;
                            dummy(noisy_idx_20) = 0 ;
                            te_distorted_20(n,:,:,:) = dummy ;  
                            
                            dummy = the_sample ;
                            dummy(noisy_idx_30) = 0 ;
                            te_distorted_30(n,:,:,:) = dummy ; 
                            
                        end
                                                                                                                                    
%                                 occ_voxels = numel(find(the_sample)) ;
%                                 occupancy_later =  occ_voxels + occupancy_later ;              
                 end                
                end
            save(noisy_file_name, 'te_distorted_20' ) ;
%             save(noisy_file_name, 'te_distorted_20' , 'te_distorted_30', 'te_distorted_50')    
            clear te_data te_distorted_20 te_distorted_50 te_distorted_30
    end       
% avg_occupancy = avg_occupancy/no_samples 
% occupancy_later = occupancy_later/no_samples