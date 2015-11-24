function out = mynormalize(in)

out = bsxfun(@minus, in, mean(in,1));
interval = max(out) - min(out);
out = bsxfun(@rdivide, out, interval);

end