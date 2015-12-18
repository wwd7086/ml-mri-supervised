clear;
close all;

load('Train.mat');
load('Test.mat');

%normalize

% Xtrain = normr(Xtrain);
% Xtrain = normc(Xtrain);

Xtrain = mynormalize(Xtrain);
Xtest = mynormalize(Xtest);

outlier_frac = 0.15;
kernel_scale = 56;
box_constraint = 40;
shrinkage = 2400;
err_rate = cross_validate_baseline_all( kernel_scale, box_constraint, ...
    Xtrain, Ytrain, outlier_frac, shrinkage)

% Get data NOT in CV and train on it
train_label = Ytrain;
train_data = Xtrain;
% 0 vs all
cur_label = zeros(size(train_label));
cur_label(train_label==0) = 1;
cv_models{1} = fitcsvm(...
    train_data, cur_label,...
    'KernelFunction', 'rbf', 'KernelScale', kernel_scale,...
    'BoxConstraint', box_constraint, ...'OutlierFraction', outlier_frac); 
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

[~, cv_score1] = predict(cv_models{1}, Xtest);
[~, cv_score2] = predict(cv_models{2}, Xtest);
[~, cv_score3] = predict(cv_models{3}, Xtest);

all_scores = [cv_score1(:,2), cv_score2(:,2), cv_score3(:,2)];

[~,ytrain] = max(all_scores,[],2);
%ytrain(ytrain==1) = 0;
%ytrain(ytrain==2) = 1;
%ytrain(ytrain==3) = 3;

writeScore = zeros(size(all_scores));
for i=1:size(ytrain,1)
    writeScore(i,ytrain(i)) = 1;
end

csvwrite('prediction.csv', writeScore);