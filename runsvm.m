clear;
close all;

load('data.mat');
models = cell(5,1);

% train 1 vs all
for i=1:5
    label=zeros(size(trainY));
    label(trainY==i)=1;
    models{i} = fitcsvm(trainX,label,'KernelFunction','rbf',...
        'KernelScale',3e5,'BoxConstraint',2000);
end

% test
scores = zeros(size(testY,1),5);
for i=1:5
    [~,score] = predict(models{i},testX);
    if i==3
        score = score-1;
    end
    scores(:,i) = score(:,2); 
end

[~,testResult] = max(scores,[],2);
err = sum(testResult==testY)/size(testY,1);