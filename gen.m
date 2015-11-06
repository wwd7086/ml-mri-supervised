clear;
close all;

load('project1data_labels.mat');
load('project1data.mat');

totalSize = size(X,1);
trainSize = int16(totalSize*0.8);
testSize = totalSize-trainSize;

ri = randperm(totalSize);
trainInd = ri(1:trainSize);
testInd = ri(trainSize+1:totalSize);

trainX = double(X(trainInd,:));
trainY = Y(trainInd,:);
testX = double(X(testInd,:));
testY = Y(testInd,:);

save('data.mat','trainX','trainY','testX','testY');
