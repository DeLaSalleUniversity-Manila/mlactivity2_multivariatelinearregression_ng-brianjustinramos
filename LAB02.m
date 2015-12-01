%LAB02

clear all; close all; clc
x = load('ex2x.dat'); 
y = load('ex2y.dat');

%Procedure 2.0
%1 Raw data
    %x1 data plot
    figure, plot(x(:,1),y,'o');
    xlabel('Living Area');
    ylabel('Housing Prices');
    title('Housing Prices vs Living Area (RAW)');

    %x2 data plot
    figure, plot(x(:,2),y,'o');
    xlabel('Number of Bedrooms');
    ylabel('Housing Prices');
    title('Housing Prices vs Number of Bedrooms (RAW)');

%2 Preprocessed data    
    m = length(x); %m = 47
    x = [ones(m, 1), x]; %x0 = 1 intercept term

    %Scaling
    mu = mean(x);
    sigma = std(x);
    x(:,2) = (x(:,2) - mu(2))./ sigma(2);
    x(:,3) = (x(:,3) - mu(3))./ sigma(3);

    
    figure, plot(x(:,2),y,'o');
    xlabel('Living Area');
    ylabel('Housing Prices');
    title('Housing Prices vs Living Area (Scaled)');

    figure, plot(x(:,3),y,'o');
    xlabel('Number of Bedrooms');
    ylabel('Housing Prices');
    title('Housing Prices vs Number of Bedrooms (Scaled)');

%Procedure 2.1
    
    %Initialization    
    theta = zeros(3,1); 

    %Selecting a learning rate
    J1=zeros(50,1);
    alpha = 0.01;

    for i = 1:50
            J1(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);
    end
    figure;
    plot(0:49,J1(1:50),'b-')


    theta = zeros(3,1); 
    J2=zeros(50,1);
    alpha = 0.03;
    for i = 1:50
            J2(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);;
    end
    hold on;
    plot(0:49,J2(1:50),'r-')


    theta = zeros(3,1); 
    J3=zeros(50,1);
    alpha = 0.1;
    for i = 1:50
            J3(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);;
    end
    hold on;
    plot(0:49,J3(1:50),'k-')


    theta = zeros(3,1); 
    J4=zeros(50,1);
    alpha = 0.3;
    for i = 1:50
            J4(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);
    end
    hold on;
    plot(0:49,J4(1:50),'y-')


    theta = zeros(3,1); 
    J5=zeros(50,1);
    alpha = 1.0;
    for i = 1:50
            J5(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);;
    end
    hold on;
    plot(0:49,J5(1:50),'c-')
    theta_final = theta


    theta = zeros(3,1); 
    J6=zeros(50,1);
    alpha = 1.3;
    for i = 1:50
            J6(i) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
            %Result of gradient update
            theta = theta - alpha .* (1/m) .* x' * ((x * theta) - y);;
    end

    hold on;
    plot(0:49,J6(1:50),'g-')
    xlabel('Number of iterations')
    ylabel('Cost J')
    legend('0.01','0.03','0.1','0.3','1.0','1.3')

%Procedure 2.2    
%Scaling
    x = load('ex3x.dat'); 
    y = load('ex3y.dat');
    x = [ones(m, 1), x]; %x0 = 1 intercept term
    price_grad_desc = dot(theta_final, [1, (1650 - mu(2))/sigma(2),...
                        (3 - mu(3))/sigma(3)]);
    theta_normal = (x' * x)\x' * y;
    house_price = dot(theta_normal, [1, 1650, 3])
    
%Procedure 2.3
    x = load('ex3x.dat'); 
    y = load('ex3y.dat');
    x = [ones(m, 1), x]; %x0 = 1 intercept term
    theta = zeros(3,1); 
    theta_closedform = ((x'*x)^-1)*(x'*y);
    house_price_closedform = dot(theta_closedform, [1, 1650, 3])
    