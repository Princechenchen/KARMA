function [ROIpoint]=gen_ROIs(model,pnum,occupied_table)
occupied_table = table2array(occupied_table);
occupied_table = occupied_table(:,1:2*model.variables.dim);
candnum = floor(pnum);
onum = size(occupied_table,1);
ROIpoint1 = lhsdesign(candnum,model.variables.dim);
ROIpoint=[];
for i=1:size(occupied_table,1)
ROIpoint0 = lhsdesign(floor(candnum/(onum+1)),model.variables.dim);
upb = occupied_table(i,2:2:end);
dob = occupied_table(i,1:2:end);
ROIpoint0 = ROIpoint0.*(upb-dob)+dob;
ROIpoint = [ROIpoint;ROIpoint0];
in_range = ROIpoint1<=upb & ROIpoint1>=dob;
ind(:,i) = all(in_range, 2);
end
globalnum = candnum-size(ROIpoint,1);
globalpoint = ROIpoint1(any(ind,2)==0,:);
if size(globalpoint,1)<globalnum
    ROIpoint = [ROIpoint;globalpoint];
else
rand_indices = randperm(size(globalpoint,1), globalnum);
ROIpoint = [ROIpoint;globalpoint(rand_indices,:)];
end
