function f = griewank(x)
% 向量化实现的Griewank函数
% 更高效，适合矩阵运算
f=1;
sumS=sum(x.^2);
pcos=1;
for i =1:6
    pcos=pcos*cos(x(i)/sqrt(i));
end
f=f+(1/4000)*sumS-pcos;
end