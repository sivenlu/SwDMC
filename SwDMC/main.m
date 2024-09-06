clear;
Dataset_name = {
    '3Sources_mv'; % 1
    'bbcsport'; % 2
    'ORL_mv1'; % 3
    'ORL_mv2'; % 4
    'WebKB'; % 5
    'Caltech101-20'; % 6
    'Caltech101-7'; % 7
    };

for i = 7
    dataset_name = [Dataset_name{i} '.mat'];
    data = load(dataset_name);

    switch(i)
        case 1
            y = data.Y;
            c= length(unique(y));
            X = normalize(data.baseCls, "norm");
            k = 51;
            r = 200;
            [predicted_label, A] = ANCMM(X', c, k, r);
            result_ANCMM = ClusteringMeasure(y, predicted_label);
            % multi view similarity matrix
            x = {A};
        case 2
            y = data.Y;
            c= length(unique(y));
            k = 188;
            r = -1;
            X1 = normalize(data.X{1}, "norm");
            X2 = normalize(data.X{2}, "norm");
            [predicted_label1, A1] = ANCMM(X1', c, k, r);
            [predicted_label2, A2] = ANCMM(X2', c, k, r);
            x = {A1, A2};
            result_ANCMM1 = ClusteringMeasure(y, predicted_label1);
            result_ANCMM2 = ClusteringMeasure(y, predicted_label2);
            result_ANCMM = (result_ANCMM1 + result_ANCMM2) / 2;
        case 3
            y = data.Y;
            c = length(unique(y));
            X1 = normalize(data.X, 'norm');
            dataset_name = [Dataset_name{i + 1} '.mat'];
            data = load(dataset_name);
            X2 = normalize(data.X, 'norm');
            
%             [predicted_label1, A1] = ANCMM(X1', c, 12, 10);
%             [predicted_label2, A2] = ANCMM(X2', c, 12, 10);
            [predicted_label1, A1] = CAN(X1', c, 10, 20);
            [predicted_label2, A2] = CAN(X2', c, 10, 20);
            x = {A1, A2};
            result1 = ClusteringMeasure(y, predicted_label1);
            result2 = ClusteringMeasure(y, predicted_label2);
%             result = (result1 + result2) / 2; 
            disp(result1);
            disp(result2);
            
        case 4
            continue;
        case 5
            y = data.gnd;
            c = length(unique(y));
            k = 20;
            r = 200;
            X1 = normalize(data.X{1}, "range");
            X2 = normalize(data.X{2}, "range");
            [predicted_label1, A1] = ANCMM(X1', c, k, r);
            [predicted_label2, A2] = ANCMM(X2', c, k, r);

%             [predicted_label1, A1] = CAN(X1', c, k, r);
%             [predicted_label2, A2] = CAN(X2', c, k, r);

            x = {A1, A2};
            result1 = ClusteringMeasure(y, predicted_label1);
            result2 = ClusteringMeasure(y, predicted_label2);
            result = (result1 + result2) / 2;
        case {6, 7}
            y = data.Y;
            c = length(unique(y));
            if i == 6
                k = 20;
                r = 200;
            else
                k = 5;
                r = 20;
            end
            % data
            X = cell(1, 6);
            predicted_label = cell(1, 6);
            % multi view similarity matrix
            x = cell(1, 6);
            result = zeros(6, 3);
            for j = 1 : 6
                X{j} = normalize(data.X{j}, "range");
                [predicted_label{j}, x{j}] = ANCMM(X{j}', c, k, r);
                result(j, :) = ClusteringMeasure(y, predicted_label{j});
            end                
    end
    [result_SwDMC, S, clusternum] = SwDMC(x, y, c);
    if clusternum ~= c
        sprintf('Can not find the correct cluster number: %d, get %d', clusternum, c);
    end
%     display(result);
    display(result_SwDMC);
%     disp(sum(S));
end
