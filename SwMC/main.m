clear;
Dataset_name = {
    '3Sources_mv'; % 1
    'bbcsport'; % 2
    'ORL_mv1'; % 3
    'ORL_mv2'; % 4
    'WebKB'; % 5
    };

for i = 5
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
            X = {A};
        case 2
            y = data.Y;
            c= length(unique(y));
            k = 188;
            r = -1;
            X1 = normalize(data.X{1}, "norm");
            X2 = normalize(data.X{2}, "norm");
            [predicted_label1, A1] = ANCMM(X1', c, k, r);
            [predicted_label2, A2] = ANCMM(X2', c, k, r);
            X = {A1, A2};
            result_ANCMM1 = ClusteringMeasure(y, predicted_label1);
            result_ANCMM2 = ClusteringMeasure(y, predicted_label2);
            result_ANCMM = (result_ANCMM1 + result_ANCMM2) / 2;
        case 3
        case 4
        case 5
            y = data.gnd;
            c= length(unique(y));
            k = 20;
            r = 200;
            X1 = normalize(data.X{1}, "range");
            X2 = normalize(data.X{2}, "range");
%             [predicted_label1, A1] = ANCMM(X1', c, k, r);
%             [predicted_label2, A2] = ANCMM(X2', c, k, r);

            [predicted_label1, A1] = CAN(X1', c, k, r);
            [predicted_label2, A2] = CAN(X2', c, k, r);

            X = {A1, A2};
            result1 = ClusteringMeasure(y, predicted_label1);
            result2 = ClusteringMeasure(y, predicted_label2);
            result = (result1 + result2) / 2;
    end
    [result_SwDMC, S, clusternum] = SwDMC(X, y, c);
%     if clusternum == c
%         disp(k);
%         disp(r);
%     end
    display(result);
    display(result_SwDMC);
%     disp(sum(S));
end
