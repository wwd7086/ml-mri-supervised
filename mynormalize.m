function out = mynormalize(in)

% h = fspecial('gaussian', [5 1], 0.5);
% in = imfilter(in, h);

% kernel = ones(3,3,3);
% kernel = kernel / numel(kernel);
% in = imfilter(in, kernel);

out = bsxfun(@minus, in, mean(in,1));
interval = max(out) - min(out);
out = bsxfun(@rdivide, out, interval);

end