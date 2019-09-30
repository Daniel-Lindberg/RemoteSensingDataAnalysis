

% A,B : Matrices containing the correlation coefficient
% r: Column matrix containing the canonical correlations
% U,V: Canonical variates/basis vectors for A,B respectively
% stat: statistics for hypothesis testing
%AUthor Daniel Lindberg 


I = double(imread('geo_1.jpeg')); 
J = double(imread('geo_2.jpeg'));
[A, B, r , u , v] = canoncorr(I(:,:),J(:,:));
figure;
plot(u(:,1),v(:,1),'.');
figure;
plot(u(:,end),v(:,end),'.');
MAD = zeros(4700, 987);
for i = 1:4700
    for j = 1:987
        MAD(i,j) =  u(i,j) - v(i,j);
    end
end 

MAD2 = MAD;
for i = 1:4700
    for j = 1:987
        MAD2(i,j) = MAD(i,j)*100;
    end
end

image(MAD2);    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





