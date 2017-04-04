
load('feat_te_10_64_256_64_.1_nll.mat')

% calculate the l2 norm between each of the test query 
% in all 2400
test_features = feat_mat_te_10_nll ;
no_classes = 10 ;

load ('mul-class_te_10_reduced.mat')
test_labels = te_labels_r ;
no_files_per_class = 12*20;

for i = 1:no_classes      
    % define a label vector of +1 and -1            
    for j = 1:no_files_per_class        
        idx = no_files_per_class* (i-1) + j ;        
         label_vec = test_labels ;
         
         label_vec(idx) = [] ;     
      label_vec(label_vec~=i) = -1 ;
      
      %query feature
       X = test_features(idx, :) ;         
       % remaining features after removing query feature
       
       dummy = test_features ;
       dummy(idx, :) = [] ;
       Y = dummy ;
        
       scores = pdist2(X,Y, 'cosine');      
       [RECALL, PRECISION, INFO] = vl_pr(label_vec', scores ) ;
       
       INFO.ap
    end    
    
end