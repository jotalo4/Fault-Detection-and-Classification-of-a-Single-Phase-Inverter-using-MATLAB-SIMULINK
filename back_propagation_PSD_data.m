% NN implementation via backpropagation using 
% one hidden layer (sigmoid) and one output layer (linear)
% For use with PSD data from 1-phase Simulink inverter with MOSFET faults
% Written by, V. Winstead
% Dec 1, 2019
%
% hidden layer fn --> logsig(n) = 1 / (1 + exp(-n))
% output layer fn --> purelin(n) = n

close all
clear

% configuration
n1 = 5; % number of neurons in layer 1 (hidden layer)
i_size = 10; % length of input vector
o_size = 4; % length of output vector
my_alpha = 0.005; % learning rate

s = rng;
% initial conditions on weights (W) and bias offsets (b)
%% adjusted starting weight and bias
W1 = randn(n1, i_size);
b1 = randn(n1, 1);
W2 = randn(o_size, n1);
b2=  randn(o_size, 1);

% W1 = -0.01 + (0.01-(-0.01)).*rand(n1, i_size);
% b1 = -0.01 + (0.01-(-0.01)).*rand(n1, 1);
% W2 = -0.01 + (0.01-(-0.01)).*rand(o_size, n1);
% b2= -0.01 + (0.01-(-0.01)).*rand(o_size, 1);

% open data file
fileID = fopen('NN_data_test_s1_s2_s3_s4_20000.txt','r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %d %d %d %d %f';
A = fscanf(fileID,formatSpec, [15 Inf]);
fclose(fileID);
A = A'; % data read in transposed
p = A(:,1:10); % PSD data only
fault_device = A(:,11:14); % fault condition for each MOSFET
my_index = length(p);  % number of iterations to train NN

    
for z = 1:my_index
    
    % compute output of hidden layer
    my_temp = W1 * p(z,:)' + b1;
    a1 = 1 ./ (1 + exp(-my_temp));
    %a1 = min(my_temp, 0)

    % compute output of output layer
    my_temp = W2 * a1 + b2;
    a2 = my_temp;

    % compute error
    my_e = fault_device(z,:)' - a2;

    % compute partial differentials
    F1 = diag((1-a1).*a1);
    F2 = diag(ones(o_size, 1));

    % compute backpropagation starting with 2nd layer
    s2 = -2 * F2 * my_e;
    s1 = F1 * W2' * s2;

    % update weights and bias values
    W2 = W2 - my_alpha * s2 * a1';
    b2 = b2 - my_alpha * s2;
    W1 = W1 - my_alpha * s1 * p(z)';
    b1 = b1 - my_alpha * s1;
    
    my_e_out(z) = norm(my_e); % store metric of errors
    
     % Average Training error
    E1(z) = my_e_out(z)/ my_index;
end


plot(E1)
xlabel('Epoch')
ylabel('Average of Training error')
legend('Average Training error')
title('model loss')


