clear all

%path = '/Users/amandaoliveira/Documents/Mestrado/Tese - Codigo/'; filename = 'test.rbm';

path = '/Users/amandaoliveira/Documents/Mestrado/Tese - Codigo/Training Outputs/Redes Treinadas/H500_nets';
filename = 'mnist_sgd-0.5_H500_CD-10_lr0.01_mBatch50_iter200_run0.rbm'; 


inputFile = sprintf('%s/%s', path, filename);

% filetext = fileread(inputFile);

fprintf('Opening RBM from file %s\n', filename)

fid = fopen(inputFile);
tline = fgetl(fid);

% while ischar(tline)
%     disp(tline)
%     tline = fgetl(fid);
% end

while tline(1) == '#'
    fprintf('\t%s\n', tline(3:end)) 
    tline = fgetl(fid);
end

tmp = regexp(tline, '\d*', 'Match');
X = str2double(tmp{1});
H = str2double(tmp{2});

fprintf( 'RBM has %i visible units and %i hidden units\n', X, H )

% Reading weights
vishid = zeros(X, H);
for i=1:H
    tline = fgetl(fid);
    tmp = regexp(tline, '[+-]?(\d+,)*\d+(\.\d*)?', 'Match');
    for j=1:X
        vishid(j,i) = str2double(tmp{j});
    end
end

% Read visible biases
visbiases = zeros(1, X);
tline = fgetl(fid);
tmp = regexp(tline, '[+-]?(\d+,)*\d+(\.\d*)?', 'Match');
for j=1:X
    visbiases(j) = str2double(tmp{j});
end


% Read hidden biases
hidbiases = zeros(1, H);
tline = fgetl(fid);
tmp = regexp(tline, '[+-]?(\d+,)*\d+(\.\d*)?', 'Match');
for i=1:H
    hidbiases(i) = str2double(tmp{i});
end


% Evaluating the Log Likelihood
makebatches;
fprintf('\nCalculating true partition function of RBM\n')

%[logZZ_true] = calculate_true_partition(vishid,hidbiases,visbiases);
%loglik_test_true = calculate_logprob(vishid,hidbiases,visbiases,logZZ_true,testbatchdata);


fprintf(1,'\nEstimating partition function by running 100 AIS runs.\n');
beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/10000:1.0];  % For H=16
%beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/100000:1.0]; % For H=500
numruns = 100;

tic
[logZZ_est, logZZ_est_up, logZZ_est_down] = ...
             RBM_AIS(vishid,hidbiases,visbiases,numruns,beta,batchdata);
%loglik_test_est = calculate_logprob(vishid,hidbiases,visbiases,logZZ_est,testbatchdata);
toc

%fprintf(1,'\nTrue log-partition function: %f\n', logZZ_true);
fprintf(1,'Estimated log-partition function (+/- 3 std): %f (%f %f)\n', logZZ_est,logZZ_est_down,logZZ_est_up);
%fprintf(1,'\nAverage log-prob on the test data: %f\n', loglik_test_true);
%fprintf(1,'Average estimated log-prob on the test data: %f\n', loglik_test_est);