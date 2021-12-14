% clear all
% id = feature('getpid');

inputFile = sprintf('tmp_%i.rbm', id); 
outputfile = sprintf('lnZ_%i.txt', id);
fprintf('Opening RBM from file %s, will save file %s\n', inputFile, outputfile);

fid = fopen(inputFile);
tline = fgetl(fid);

while tline(1) == '#'
    % fprintf('\t%s\n', tline(3:end)) 
    tline = fgetl(fid);
end

tmp = regexp(tline, '\d*', 'Match');
X = str2double(tmp{1});
H = str2double(tmp{2});

% fprintf( 'RBM has %i visible units and %i hidden units\n', X, H )

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
fclose(fid);

% AIS Setup
beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/100000:1.0]; % For H=500
numruns = 100;

% makebatches;
[logZZ_est, logZZ_est_up, logZZ_est_down] = ...
             RBM_AIS(vishid,hidbiases,visbiases,numruns,beta); %,batchdata);

fid=fopen(outputfile,'w');
fprintf(fid, '%12.8f\n', logZZ_est);
fclose(fid);

% fprintf('Got Z estimate: %12.8f\n', logZZ_est);