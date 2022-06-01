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

% initial conditions on weights (W) and bias offsets (b)
W1 = randn(n1, i_size);
b1 = randn(n1, 1);
W2 = randn(o_size, n1);
b2 = randn(o_size, 1);

% open data file
fileID = fopen('NN_data_test_s2_1000.txt','r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %d %d %d %d %f';
A = fscanf(fileID,formatSpec, [15 Inf]);
fclose(fileID);
A = A'; % data read in transposed
p = A(:,1:10); % PSD data only
fault_device = A(:,11:14); % fault condition for each MOSFET
my_index = length(p);  % number of iterations to train NN

% complete my_index iterations to train NN
for z = 1:my_index
    
    % compute output of hidden layer
    my_temp = W1 * p(z,:)' + b1;
    a1 = 1 ./ (1 + exp(-my_temp));

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
end

plot(my_e_out)
xlabel('Epoch')
ylabel('Average of Training error')
legend('Training error')
title('model loss')

