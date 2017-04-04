
% script to convert asci feature files to feature matrix
no_class = 40 ;
no_pose = 12 ;
% tr_samples = no_class*no_pose*; % for 10 classes
% te_samples = 2400; % for 10 classes

if 0
    tr_samples = 38196 ;
    te_samples = 9600 ;
else
    tr_samples = 9600 ;
    te_samples = 2400 ;
end

desc_dims = 6912 ;

fid = fopen('tr_feats_10_r_64_256_nll_ft.asc','rt');

 the_sample = textscan(fid, '%f', 'HeaderLines',17);
 fclose(fid);
 the_sample = the_sample{1} ;
  
  feat_mat_tr_10_nll= reshape(the_sample, [tr_samples, desc_dims]) ;
  save('./3d_desc_files/feat_tr_10_64_256_64_.1_nll_ft.mat','feat_mat_tr_10_nll' ) % rewrite this at the top to reduce mistakes