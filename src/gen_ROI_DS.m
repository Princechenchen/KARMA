function [ROIpoint,idx]=gen_ROI_DS(model,pnum,occupied_table)
occupied_table = table2array(occupied_table); occupied_table = occupied_table(:,1:end-2);
candnum = floor(pnum);
ROIpoint0 = model.candidate;
ROIpoint=[];
for i=1:size(occupied_table,1)
upb = occupied_table(i,2:2:end);
dob = occupied_table(i,1:2:end);
in_range = ROIpoint0<=upb & ROIpoint0>=dob;
ind(:,i) = all(in_range, 2);
end
ROIpoint = ROIpoint0(any(ind,2),:);
idx = find(any(ind,2)==1); idx2 = find(any(ind,2)~=1);
globalnum = floor(candnum*0.1);
globalpoint = ROIpoint0(any(ind,2)==0,:);
if size(globalpoint,1)<globalnum
    ROIpoint = [ROIpoint;globalpoint];
    idx = [idx;idx2];
else
rand_indices = randperm(size(globalpoint,1), globalnum);
ROIpoint = [ROIpoint;globalpoint(rand_indices,:)];
idx = [idx;idx2(rand_indices)];
end
