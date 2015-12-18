clear;
close all;

load Train.mat;
load Test.mat;

addpath(genpath('DeepLearnToolbox'));

%% preprocess the data
Xtrain = mynormalize(Xtrain);
Xtest = mynormalize(Xtest);

%%  Dimensin reduction using autoencoder
rand('state',0);
aesize = [5903 3000 1000];
sae = saesetup(aesize);
% layers
for i=1:numel(aesize)-1
    sae.ae{i}.activation_function       = 'sigm';
    sae.ae{i}.learningRate              = 0.05;
    sae.ae{i}.inputZeroMaskedFraction   = 0;
end
% optimize
opts.numepochs =  20;
opts.batchsize = 100;
Xall = [Xtrain;Xtest];
sae = saetrain(sae, Xall(1:1500,:), opts);
%load sae.mat;

% global optimize
nng = nnsetup([aesize(1:end-1), fliplr(aesize)]);
for i=1:numel(aesize)-1
    nng.W{i} = sae.ae{i}.W{1};
    nng.W{numel(aesize)+2-i} = sae.ae{i}.W{2};
end
nng.activation_function = 'sigm';
nng.learningRate = 0.05;
opts.numepochs =  100;
nng = nntrain(nng, Xall(1:1500,:), Xall(1:1500,:), opts);
save('nng.mat','nng');

% Use the SDAE to initialize a FFNN
nn = nnsetup(aesize);
nn.activation_function              = 'sigm';
% layers
for i=1:numel(aesize)-1
    nn.W{i} = nng.W{i};
end

%% compute reduced Xtrain
nn = nnff(nn, Xtrain, zeros(size(Xtrain,1), nn.size(end)));
XtrainR = nn.a{end};
nn = nnff(nn, Xtest, zeros(size(Xtest,1), nn.size(end)));
XtestR = nn.a{end};
save('TrainR.mat','XtrainR','XtestR');

%% train and cross validate on svm
outlier_frac = 0.15;
kernel_scale = 56;
box_constraint = 40;
shrinkage = 2400;
err_rate = cross_validate_baseline_all( kernel_scale, box_constraint, ...
    XtrainR, Ytrain, outlier_frac, shrinkage)
