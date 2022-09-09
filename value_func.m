function J = value_func(x,u,Q,R,N)
%Performance Index
for i = 1:N
   M1(:,i) = x(:,i);
   M2(:,i) = M1(:,i)'*M1(:,i)*Q+u(:,i)*u(:,i)'*R;
   J(i) = 0.5*sum(M2(:,i));
end