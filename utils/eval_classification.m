

for i =1:2
% script to convert asci feature files to feature matrix
no_class = 40 ;
no_pose = 12 ;
desc_dims = 512 ;
% tr_samples = no_class*no_pose*; % for 10 classes
% te_samples = 2400; % for 10 classes
% rewrite this at the top to reduce mistakes

tr_samples = 38196 ;
    te_samples = 9600 ;
%      tr_samples = 9600 ;
%     te_samples = 2400 ;
if i==1
    
    fid = fopen('tr_feats_40_r_64_256_nll_ft_no_drop_512.asc','rt');

 the_sample = textscan(fid, '%f', 'HeaderLines',17);
 fclose(fid);
 the_sample = the_sample{1} ;
  
  feat_mat_tr_40_6912= reshape(the_sample, [tr_samples, desc_dims]) ;
  tr_feat_str = './feat_tr_40_64_256_64_.001_ft_no_drop_512.mat' ;
  save(tr_feat_str,'feat_mat_tr_40_6912' ) 
else
   
    fid = fopen('te_feats_40_r_64_256_nll_ft_no_drop_512.asc','rt');

 the_sample = textscan(fid, '%f', 'HeaderLines',17);
 fclose(fid);
 the_sample = the_sample{1} ;
  
 te_feat_str = './feat_te_40_64_256_64_.001_ft_no_drop_512.mat' ;
  feat_mat_te_40_6912= reshape(the_sample, [te_samples, desc_dims]) ;
  save(te_feat_str,'feat_mat_te_40_6912' ) 
end
  
  end
%% load the training data

load(tr_feat_str)
train_features = feat_mat_tr_40_6912 ; % for libsvm, feat_mat should be N by D matrix

% %% evaluate performance on test set
load(te_feat_str)

test_features = feat_mat_te_40_6912 ;

load ('mul-class_tr_40_reduced.mat')
train_labels = tr_labels_r ;

ndata = numel(train_labels) ; 
% shuffle your data before feeding to the cross validation
rand_idx = randperm(ndata) ;

train_features = train_features(rand_idx, :) ;
train_labels = train_labels(rand_idx) ;

load ('mul-class_te_40_reduced.mat')
test_labels = te_labels_r ;

cv_do = 0 ;

if cv_do
    
% using cross-validation

Nlambdas                = 3;
lambda_range            = [.1,1,10];
% for each of those lambdas

for i=1:Nlambdas   
    lambda = lambda_range(i);    
    % perform K-fold cross validation
    K = 5;   
    accuracy_cell  =zeros(1,5);
    fprintf('lambda = %.4f  [%i out of %i]\n',lambda,i,Nlambdas);
        
    for validation_run=1:K
        
        fprintf('.');
        %% TEMPLATE FOR CROSS-VALIDATION CODE       
        %split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
            split_data(train_features,train_labels,ndata,K,validation_run);
                         
                cmd = sprintf('-t 0 -c %f', lambda)  ;  
                
                model = svmtrain(trset_labels, trset_features, cmd);
                [predict_label, accuracy, dec_values] = svmpredict(vlset_labels, vlset_features, model); % test the training data                      
                 
        accuracy_cell(1,validation_run) = accuracy(1);
    end
    
    fprintf(' \n');
    %The cross-validation error is the mean of the error
    cv_accuracy(i)=mean(accuracy_cell,2);
end


[~,cv_lambda_idx] = max(cv_accuracy) ;
cv_lambda = lambda_range(cv_lambda_idx) ;
end

cv_lambda = 100 ;  % on sumatra it was for 64 and lambda was .01

cmd = sprintf('-t 0 -c %f', cv_lambda)  ;
  model_cv = svmtrain(train_labels, train_features, cmd);  
   


% nerrors_test = your_code_goes_here;

[predict_te_label, te_accuracy, dec_values] = svmpredict(test_labels, test_features, model_cv); % test the training data
% 
