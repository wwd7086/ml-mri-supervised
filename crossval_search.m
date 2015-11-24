%% Cross-validate loop

load('Train.mat');
load('Test.mat');

scale_s=50;
scale_p=10;
scale_c=10;

box_s=40;
box_p=2;
box_c=10;

err_rate =[]; %zeros(scale_c*box_c,3);
for s=1:scale_c
   box_err = zeros(box_c,3);
   parfor b=1:box_c
       scale = scale_s + (s-1)*scale_p;
       box = box_s + (b-1)*box_p;
       
       err = cross_validate_baseline(scale, box, Xtrain, Ytrain);
       fprintf('%f %f %f\n', scale, box, err);
       
       box_err(b,:) = [scale,box,err];
       %err_rate((s-1)*box_c + b,:) = [scale,box,err];
   end
   err_rate = [err_rate; box_err];
end