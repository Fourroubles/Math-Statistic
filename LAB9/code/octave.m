pkg load interval
addpath(genpath('./m'))
addpath(genpath('./data'))

format long g

%data = csvread("channel_1.csv")
data = csvread("channel_2.csv")
# remove first rows
%data(1,:) = []
#leave only mV values
data_mv = data(:,1)
# get N values
data_n = transpose(1:length(data_mv))

# get Epsilon
data_eps = 1e-4

% setup a problem
data_X = [ data_n.^0 data_n ];
data_inf_b = data_mv - data_eps
data_sup_b = data_mv + data_eps
[data_tau, data_w] = L_1_minimization(data_X, data_inf_b, data_sup_b);

fileID = fopen('ch2.txt','w');
%fileID = fopen('data/ch2.txt','w');
fprintf(fileID,'%g %g\n', data_tau(1), data_tau(2));
for c = 1 : length(data_w)
  fprintf(fileID, "%g\n", data_w(c));
end
fclose(fileID);
