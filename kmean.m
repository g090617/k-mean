% Class: EE133A
% Date: Oct 3rd 2016
% HW1.16
% Name: Haoran Zhang
% UID: 804586710
%
%%
%Load the file  and declare constants and variables
Num_Clusters = 20;
Img_Size = 784;
Norm_of_Diff = zeros(1,20);
J_Clust = 0;
J_Clust_Prev = 1;
load mnist_train.mat

%change digits from 784x60000 to 784x10000
digits = digits(:,1:10000);

%assign random group to each vector from 1 to 20
group = randi(Num_Clusters,1,10000);

%Z is group representitives 783x20
Z = zeros(Img_Size , Num_Clusters);
%%
%for each iteration when (J_Clust_Prev-J_Clust)/J_Clust>10e-5 
%calculate the mean from each group 1 to 20
%for each vector
%calculate the norm to all 20 representitives
%choose the minimum norm and assign group index
%to that digit

while abs((J_Clust_Prev-J_Clust)/J_Clust) > 10e-5
    J_Clust_Prev = J_Clust
    for ii = (1 : Num_Clusters)
        I = find(group == ii);
        G = digits(:,I);
        Z(:,ii) = mean(G,2);
    end
    
    for ii = (1 : 10000)
        %for kk =(1 : 20)
        %   Norm_of_Diff(kk) = norm(V(:,kk))^2;
        %end
        V = repmat(digits(:,ii),1,20)-Z;      %Instead of using for loops
        Temp = V.*V;                          %matrix operation is much faster
        Norm_of_Diff = sum(Temp,1);
        [Val,group(ii)] = min(Norm_of_Diff);
        J_Clust = J_Clust +Val;
    end
    J_Clust = J_Clust/10000;
end
%%
Extended_Digits = repmat(digits(:,:),2,20);
Extended_Z = repmat(Z(:,:),2,10000);
Permuted_Z = permute(Extended_Z,[1 3 2]);
V = Extended_Digits - Permuted_Z;

%%
%display the final 20 representitives
for kk = 1:20
    subplot(4,5,kk)
    imshow(reshape(Z(:,kk),28,28));
end
