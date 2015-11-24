function [ num_errors ] = getErrors( cv_label, label_a, label_b, ...
    ytrain_a, ytrain_b )
% Find the number of label_a in ytrain_a but not in cv_label, and vice
% versa. Then find the number of label_b in ytrain but not in cv_label, and
% vice versa. Combine the two to get the number of wrongly labeled items.

    cv_label_a = find(cv_label==label_a);
    wrong_a = numel(setdiff(ytrain_a, cv_label_a)) + numel(setdiff(...
        cv_label_a, ytrain_a));
    
    cv_label_b = find(cv_label==label_b);
    wrong_b = numel(setdiff(ytrain_b, cv_label_b)) + numel(setdiff(...
        cv_label_b, ytrain_b));
    
    num_errors = wrong_a + wrong_b;    
end

