clear;
addpath(genpath(pwd));
Dataset_name = { % index    number of objects
    '3sources'; % 1         169
    'bbcsport'; % 2         544
    'ORL'; % 3              400
    'NUS'; % 4              2400
    'scene'; % 5            2688   
    'COIL20'; % 6           1440
};
lambda = [0.001, 0.1, 0.01, 0.01, 1, 0.001];
k = [50, 150, 16, 2000, 2000, 40]; % 50
r = [-1, -1, -1, -1, -1, -1];
obj_value_list = zeros(size(Dataset_name, 1), 30);
result_list_SFDMC = zeros(size(Dataset_name, 1), 3);
for i = 1:6
    dataset_name = [Dataset_name{i} '.mat'];
    data = load(dataset_name);
    
    Y = data.Y;
    c = length(unique(Y));
    % data cell
    X = data.X;
    % number of views
    m = size(X, 2);

    predicted_label = cell(1, m);
    % multi view similarity matrix
    % A{i} is the i-th view's similarity matrix
    A = cell(1, m);
    % result(i, :) is the i-th view's [ACC, NMI, PUR]
    result = zeros(m, 3);
    % get every view's similarity matrix by CAN(no rank contraint version)
    % X{j} is the i-th view's data
    for j = 1 : m
        X{j} = normalize(X{j}, "range");
        A{j} = CAN_no_rank_constraint(X{j}', k(i), r(i));
    end 
    
    [result_list_SFDMC(i, :), S, clusternum] = SwDMC(A, Y, c, lambda(i));

    disp(result_list_SFDMC(i, :));
%     disp(sum(S, 2));
end
rmpath(genpath(pwd));