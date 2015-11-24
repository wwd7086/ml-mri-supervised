function [ err_rate ] = cross_validate_baseline_all( kernel_scale, ...
    box_constraint, Xtrain, Ytrain, outlier_frac, shrinkage)
    %% Cross-validate
    % Can only use outlier fraction OR shrinkage, but not at the same time.
        
    total_train = size(Xtrain,1);
    rand_indexes = randperm(total_train);
    num_folds = 10;

    num_train = floor(total_train / num_folds); % round down to integer

    cv_models = cell(3,1);

    total_err_rate = 0;
    % split into training/testing
    for i=1:num_folds
    %for i=1:2
        % Get num_train number of indexes out of rand_indexes
        idx_start = (i-1) * num_train + 1;
        idx_end = i * num_train;
        idx_arr = rand_indexes(idx_start:idx_end);
        % Create the cross-validation hold out with the indexes
        test_data = Xtrain(rand_indexes(idx_arr),:);
        test_label = Ytrain(rand_indexes(idx_arr));

        % Get data NOT in CV and train on it
        train_idx = setdiff(rand_indexes, idx_arr);
        train_data = Xtrain(rand_indexes(train_idx),:);
        train_label = Ytrain(rand_indexes(train_idx));

        % 0 vs all
        cur_label = zeros(size(train_label));
        cur_label(train_label==0) = 1;
        cv_models{1} = fitcsvm(...
            train_data, cur_label,...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint, ...'OutlierFraction', outlier_frac );
            'Shrinkage', shrinkage, 'GapTolerance', 1e-2);
        % 1 vs all
        cur_label = zeros(size(train_label));
        cur_label(train_label==1) = 1;
        cv_models{2} = fitcsvm(...
            train_data, cur_label,...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint, ...'OutlierFraction', outlier_frac);
            'Shrinkage', shrinkage, 'GapTolerance', 1e-2);
        % 3 vs all
        cur_label = zeros(size(train_label));
        cur_label(train_label==3) = 1;
        cv_models{3} = fitcsvm(...
            train_data, cur_label,...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint, ...'OutlierFraction', outlier_frac);
            'Shrinkage', shrinkage, 'GapTolerance', 1e-2);

        [~, cv_score1] = predict(cv_models{1}, test_data);
        [~, cv_score2] = predict(cv_models{2}, test_data);
        [~, cv_score3] = predict(cv_models{3}, test_data);

        all_scores = [cv_score1(:,2), cv_score2(:,2), cv_score3(:,2)];

        [~,ytrain] = max(all_scores,[],2);
        ytrain(ytrain==1) = 0;
        ytrain(ytrain==2) = 1;
        ytrain(ytrain==3) = 3;

        total_err_rate = total_err_rate + sum(abs(ytrain - test_label) > 0) / num_train;     
    end

    err_rate = total_err_rate / num_folds;

end