Nsim = 50; %simulation steps, change to whatever value
x = zeros(Nsim,2)'; %store states
u = zeros(Nsim-1,1)'; %store inputs
c = zeros(Nsim-1,2)'; %store varying (state, input dependent) params
x(:,1) = 2*randn(2,1); %initial state, randomized

A = [1 2;-2 1]; % model matrices
B = [0.5;1];
%K = -dlqr(A,B,eye(2),0.001);
K = -place(A,B,[0.9 0.8]'); %control policy to stabilized a LTI system

for i=1:Nsim-1
    
   u(i) = K*x(:,i); %policy is u_k=K*x_k
   c(:,i) = [0.1*sin(x(2,i));0.1*cos(x(1,i))]*u(i)*0.25; %c(x,u)
   x(:,i+1) = A*x(:,i)+B*u(i)+c(:,i); %x' = Ax + Bu + c(x,u), system model

    
end

figure(1);plot(x(1,:),x(2,:),'-o');hold all;grid on;
figure(2);
subplot(211);
plot(x(1,:));hold all;grid on;
subplot(212);
plot(x(2,:));hold all;grid on;