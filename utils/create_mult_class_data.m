% this script creates the multi-class data for 10 classes and stores in a
% new folder..it also creates random noise and slicing nosie test set

structured_dataset = 0;
multi_class_data = 0 ;
data_path = '/BS/deep_3d/work/deep_3d/fcn_3D/Data/';
% param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};

slicing_noise_data = 0 ;
classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};

% param finish
save_path = [data_path, 'mul-class/mul-class_tr_10.mat'] ;

files = dir(data_path) ;
tr_data_10_class = [] ;

for i = 1 : length(files)              
              % you only want to select the files that start with class
              % name and end with _te          
              % just concat them randomly
                if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || strcmp(files(i).name(1:5), 'rand_') ||strcmp(files(i).name(1:5), 'dist_') ||~strcmp(files(i).name(end-6:end), '_tr.mat')
                    continue;
                end          
                 str_len = length(files(i).name) ;
                class_name = files(i).name(1:str_len-7);
                assert(any(ismember(classnames,class_name ))) ;          
                 
                load(files(i).name)   % load the original file
                dims = size(tr_data);
                tr_data_10_class = [tr_data_10_class; tr_data] ;
                no_samples = dims(1) 
end

save(save_path, 'tr_data_10_class');
%         bb = randperm(30,1);
%         cc = randperm(30,1) ;
%         dd = randperm(30,1) ;
        
        % select a random cube
%         if bb+5>30
%             b = [bb,bb-1, bb-2, bb-3, bb-4, bb-5];
%         else
%             b = [bb,bb+1, bb+2, bb+3, bb+4, bb+5];
%         end
%         
%         if cc+5>30
%             c = [cc,cc-1, cc-2, cc-3, cc-4, cc-5];
%         else
%             c = [cc,cc+1, cc+2, cc+3, cc+4, cc+5];
%         end
%         
%         if dd+5>30
%             d = [dd,dd-1, dd-2, dd-3, dd-4, dd-5];
%         else
%             d = [dd,dd+1, dd+2, dd+3, dd+4, dd+5 ];
%         end
%         the_sample(:,b,c,d) = 0;
%         
%         
%         occ_voxels = numel(find(the_sample)) 
%         occupancy_later =  occ_voxels + occupancy_later ;
%         
%         display 'next iteration' 
%         te_distorted_10(i,:,:,:) = the_sample ;
%          
%      end
% end