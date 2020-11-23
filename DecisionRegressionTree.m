% 01: calculate the standard deviation of target
% 02: split the dataset into different features, then calculate the standard deviation for each
% StandardDeviationReduction(T,X)=S(T)-S(T,X)
% 03: Choose feature with largest standard deviation reduction as decision node
% 04: Divide dataset based on the values of the selected feature
% 05: Branch set with CV more than 20 needs further splitting
% 06: Run recursively on the non-leaf branches
    
function untitled2
    function [p,sd,sub1,sub2] = split(x, y)
        x_min = min(x);
        x_max = max(x);
        inc = (x_max - x_min)/1000; % length of increament
        sd = std(y);
        p = x_min;
        sub1 = find(x>p);
        sub2 = find(x<=p);
        for pt = (x_min+inc):inc:x_max
            sub_index1 = find(x>pt);
            sub_index2 = find(x<=pt);
            sd1_temp = std(y(sub_index1));
            sd2_temp = std(y(sub_index2));
            num_sub1 = length(sub_index1);
            num_sub2 = length(sub_index2);
            sd_temp = (num_sub1*sd1_temp+num_sub2*sd2_temp)/length(x);
            if(sd_temp<sd)
                p = pt;
                sd = sd_temp;
                sub1 = sub_index1;
                sub2 = sub_index2;
            end    
        end
    end

    function buildtree(X,y,depth,flag)
        n = length(y);
        min_node = 100;
        CV = (nanstd(y)/nanmean(y))*100;
        fprintf('DepthofNode = %d. NodeValue = %f. FlagSign = %d.\n', depth, mean(y), flag);
        if(n>=min_node)
            [index,p,sd,sub1,sub2]=buildnode(X,y);
            if CV<20
                return;
            end
            y_left=y(sub1); y_right=y(sub2);
            X_left=X(sub1,:); X_right=X(sub2,:);
            fprintf('Column = %d. SplitValue = %f. StandardDeviation = %f.\n', index, p, sd);
            depth = depth+1;
            buildtree(X_left, y_left, depth, 1);  
            buildtree(X_right, y_right, depth, 0);
            depth = depth-1;
        end
    end

    function [ index, p, sd, sub1, sub2 ] = buildnode( X, y )
        [x_row, x_col] = size(X);
        sd = std(y);
        sd = ceil(sd);
        index = 0;
        for i = 1:x_col
            [p_i,sd_i,sub1_i,sub2_i]=split(X(:,i),y);
            if(sd_i<sd)
                sd = sd_i;
                index = i;
            end
        end
        [p,sd,sub1,sub2]=split(X(:,index),y);
    end

    clear
    clc

    data=readtable('Metro_Interstate_Traffic_Volume.csv');
    % train: 70%, test: 30%
    cv = cvpartition(size(data,1),'HoldOut',0.3);
    idx = cv.test;
    % Separate to training and test data
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);

    Xy=dataTrain;
    traffic_volume=Xy(:,end);
    traffic_volume=table2array(traffic_volume);

    X=dataTrain;
    holiday	=X(:,1);
    temp	=X(:,2);
    rain_1h	=X(:,3);
    snow_1h	=X(:,4);
    clouds_all	=X(:,5);
    weather_main	=X(:,6);
    weather_description	=X(:,7);
    date_time	=X(:,8);

    X=[temp, rain_1h, snow_1h, clouds_all];
    X=table2array(X);
    buildtree(X,traffic_volume,1,1);

end
