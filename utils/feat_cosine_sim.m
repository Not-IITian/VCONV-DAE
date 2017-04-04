

% load feature desc file for chair%
% also load others file  %


load 

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
              
end

