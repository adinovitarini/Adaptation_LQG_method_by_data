function [P,K,G] = value_iteration(A,N,B,Q,R)
%Value Iteration Algorithm
P = zeros(size(A,1),size(A,1),N);
P_new = P;
K = zeros(N,size(A,1));
K_new = K;
[Pdare,Kdare] = idare(A,B,Q,R);
for i = 1:N
   P_new(:,:,i) = Q+K(i,:)'*R*K(i,:)+(A-B*K(i,:))'*P(:,:,i)*(A-B*K(i,:));
   K_new(i,:) = inv(R+B'*P_new(:,:,i)*B)*B'*P_new(:,:,i)*A;
   P(:,:,i+1) = P_new(:,:,i);
   G(i) = norm(K(i,:)-Kdare);
   K(i+1,:) = K_new(i,:);
end
%output 
K = K_new;
P = P_new;
G = G;