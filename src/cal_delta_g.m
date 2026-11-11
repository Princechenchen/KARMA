function delta_g=cal_delta_g(gp,gx,randomPoints,Input)
delta_g=zeros(size(randomPoints,1),1);
idx = zeros(size(randomPoints,1), 1);
R = pdist2(Input, randomPoints);
[~, index] = min(R(:,:));
idx(:, 1) = index';
for i=1:size(gp,3)
for j =1:size(gp,1)
        sum_g=dot(gp(j,:,i),gx(idx(j),:,i))./(dot(gx(idx(j),:,i),gx(idx(j),:,i))+ 1e-200).*gx(idx(j),:,i);
        delta_g(j,i)=(norm(gp(j,:,i)-sum_g,2));
end
end
