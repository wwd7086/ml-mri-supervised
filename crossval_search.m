%% Cross-validate loop

load('Train.mat');
load('Test.mat');

Xtrain = mynormalize(Xtrain);
Xtest = mynormalize(Xtest);

outlier_frac = 0.15;
total_shrink = 2000;

shrink_s=500;
shrink_p=100;
shrink_c=2;

scale_s=30;
scale_p=8;
scale_c=10;

box_s=35;
box_p=2;
box_c=10;

err_rate =[]; %zeros(scale_c*box_c,3);
for s=1:scale_c
    for h=1:shrink_c
       %box_err = zeros(box_c,4);
       box_err = zeros(shrink_c*box_c,4);
       parfor b=1:box_c
           scale = scale_s + (s-1)*scale_p;
           box = box_s + (b-1)*box_p;
           shrinkage = shrink_s + (h-1)*shrink_p;

           err = cross_validate_baseline_all(scale, box, Xtrain, Ytrain, outlier_frac, shrinkage);
           fprintf('%f %f %f\n', scale, box, err);

           box_err(b,:) = [scale,box,shrinkage,err];
           %err_rate((s-1)*box_c + b,:) = [scale,box,err];
       end
       err_rate = [err_rate; box_err];
    end
end