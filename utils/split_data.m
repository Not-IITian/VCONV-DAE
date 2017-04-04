function  [training_set_features,training_set_targets,validation_set_features,validation_set_targets] = split_data(inputs,outputs,nsamples,K,validation_run);
%%  [training_set_inputs,training_set_targets,validation_set_features,validation_set_targets] = split_data(inputs,outputs,nsamples,K,validation_run);


%% size of validation set 
nvalidation_set      = floor(nsamples/K);

%% pick indexes for the data that will be used as validation set
validation_indexes    = nvalidation_set*(validation_run -1) + [1:nvalidation_set];
training_indexes      = setdiff([1:nsamples],validation_indexes);

training_set_features = inputs(training_indexes, :);
training_set_targets  = outputs(training_indexes);

validation_set_features = inputs(validation_indexes, :);
validation_set_targets  = outputs(validation_indexes);
