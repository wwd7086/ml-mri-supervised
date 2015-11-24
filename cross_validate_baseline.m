function [ err_rate ] = cross_validate_baseline( kernel_scale, box_constraint, Xtrain, Ytrain)
    %% Cross-validate
        
    total_train = size(Xtrain,1);
    rand_indexes = randperm(total_train);
    num_folds = 10;

    num_train = floor(total_train / num_folds); % round down to integer

    cv_models = cell(3,1);
%     kernel_scale = 100;
%     box_constraint = 50;

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

        % 0 1
        cv_models{1} = fitcsvm(...
            [train_data(train_label==0,:); train_data(train_label==1,:)],...
            [train_label(train_label==0); train_label(train_label==1)],...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint);
        % 0 3
        cv_models{2} = fitcsvm(...
            [train_data(train_label==0,:); train_data(train_label==3,:)],...
            [train_label(train_label==0); train_label(train_label==3)],...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint);
        % 1 3
        cv_models{3} = fitcsvm(...
            [train_data(train_label==1,:); train_data(train_label==3,:)],...
            [train_label(train_label==1); train_label(train_label==3)],...
            'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
            'BoxConstraint', box_constraint);

        [~, cv_score1] = predict(cv_models{1}, test_data);
        [~, cv_score2] = predict(cv_models{2}, test_data);
        [~, cv_score3] = predict(cv_models{3}, test_data);

        score0 = cv_score1(:,1) + cv_score2(:,1);
        score1 = cv_score1(:,2) + cv_score3(:,1);
        score3 = cv_score2(:,2) + cv_score3(:,2);
        all_scores = [score0, score1, score3];

        [~,ytrain] = max(all_scores,[],2);
        ytrain(ytrain==1) = 0;
        ytrain(ytrain==2) = 1;
        ytrain(ytrain==3) = 3;
    %     ytrain0 = find(ytrain==0);  % Ytrain with label 0
    %     ytrain1 = find(ytrain==1);  % Ytrain with label 1
    %     ytrain3 = find(ytrain==3);  % Ytrain with label 3

        total_err_rate = total_err_rate + sum(abs(ytrain - test_label) > 0) / num_train;

    %     cv1_0 = find(cv_label1==0); % cv1 with label 0
    %     % Number of label 0 in ytrain but not in cv1, and vice versa
    %     wrong0 = numel(setdiff(ytrain0, cv1_0)) + numel(setdiff(cv1_0, ytrain0));    
    %     cv1_1 = find(cv_label1==1); % cv1 with label 1
    %     % Number of label 1 in ytrain but not in cv1, and vice versa
    %     wrong1 = numel(setdiff(ytrain1, cv1_1)) + numel(setdiff(cv1_1, ytrain1));
    %     wrong0 + wrong1
    %     wrong0 + wrong1 / num_train

    %     model1_err = getErrors(cv_label1, 0, 1, ytrain0, ytrain1)    
    %     model1_err_rate = model1_err / num_train    
    %     
    %     model2_err = getErrors(cv_label2, 0, 3, ytrain0, ytrain3)
    %     model2_err_rate = model2_err / num_train
    %     
    %     model3_err = getErrors(cv_label3, 1, 3, ytrain1, ytrain3)
    %     model3_err_rate = model3_err / num_train
    % 
    %     test_label0 = find(test_label==0);
    %     test_label1 = find(test_label==1);
    %     test_label3 = find(test_label==3);
    %     
    %     wrong0 = numel(setdiff(ytrain0, test_label0)) + numel(setdiff(test_label0, ytrain0))
    %     wrong1 = numel(setdiff(ytrain1, test_label1)) + numel(setdiff(test_label1, ytrain1))
    %     wrong3 = numel(setdiff(ytrain3, test_label3)) + numel(setdiff(test_label3, ytrain3))

        % Check how many labels are wrong           
    end

    err_rate = total_err_rate / num_folds;

end