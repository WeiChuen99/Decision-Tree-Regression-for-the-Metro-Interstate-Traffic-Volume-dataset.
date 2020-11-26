% 01: calculate the standard deviation of target
% 02: split the dataset into different features, then calculate the standard deviation for each
% StandardDeviationReduction(T,X)=S(T)-S(T,X)
% 03: Choose feature with largest standard deviation reduction as decision node
% 04: Divide dataset based on the values of the selected feature
% 05: Branch set with standard deviation > 0 needs further splitting
% 06: Run recursively on the non-leaf branches
    
function DecisionRegressionTree
    
    % Main starts here
    clear
    clc
    data=readtable('Metro_Interstate_Traffic_Volume.csv');
    
    % Train: 70%, Test: 30%
    cv = cvpartition(size(data,1),'HoldOut',0.3);
    idx = cv.test;
   
    % Separate to training and test data
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);

    % Data processing 
    Xy=dataTrain;
    traffic_volume=Xy(:,end);
    traffic_volume=table2array(traffic_volume);

    X=dataTrain;
    holiday	=X(:,1);
    holiday = table2array(holiday);
    holiday = categorical(holiday);
    holiday = array2table(double(holiday));
    holiday.Properties.VariableNames = {'holiday'};

    temp	=X(:,2);
    rain_1h	=X(:,3);
    snow_1h	=X(:,4);
    clouds_all	=X(:,5);
    
    weather_main=X(:,6);
    weather_main = table2array(weather_main);
    weather_main = categorical(weather_main);
    weather_main = array2table(double(weather_main));
    weather_main.Properties.VariableNames = {'weather_main'};
    
    weather_description	=X(:,7);
    weather_description = table2array(weather_description);
    weather_description = categorical(weather_description);
    weather_description = array2table(double(weather_description));
    weather_description.Properties.VariableNames = {'weather_description'};
   
    date_time	=X(:,8);
    date_time = table2array(date_time);
    date_time = datenum(date_time);
    date_time = array2table(double(date_time));
    date_time.Properties.VariableNames = {'date_time'};

    X2=[holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time];
    X2 = table2array(X2);
    
    % Build Regression Tree
    trees = DecisionTreeLearning(X2,traffic_volume,1,1);
    DrawDecisionTree(trees);
    %PrintTree(tree, 'root');
    %tree2 = fitrtree(X2,traffic_volume);
    %view(tree2,'Mode','graph');
    
    % Sub function starts here
    function [best_threshold,std_dev,subleft,subright] = split(x, y)
        x_min = min(x);
        x_max = max(x);
        inc = (x_max - x_min)/1000; % length of increament
        std_dev = std(y);
        best_threshold = x_min;
        subleft = find(x>best_threshold);
        subright = find(x<=best_threshold);
        for pt = (x_min+inc):inc:x_max
            sub_index_left = find(x>pt);
            sub_index_right = find(x<=pt);
            sd1_temp = std(y(sub_index_left));
            sd2_temp = std(y(sub_index_right));
            num_sub1 = length(sub_index_left);
            num_sub2 = length(sub_index_right);
            sd_temp = (num_sub1*sd1_temp+num_sub2*sd2_temp)/length(x);
            if(sd_temp<std_dev)
                best_threshold = pt;
                std_dev = sd_temp;
                subleft = sub_index_left;
                subright = sub_index_right;
            end    
        end
    end

    function  [tree]= DecisionTreeLearning(X,y,depth,flag)
        % Create tree node
        tree = struct('op','','kids',[],'class','','num',0);
        
        n = length(y);
        min_node = 100;
        sigma =0.00001;     

        fprintf('DepthofNode = %d. NodeValue = %f. FlagSign = %d.\n', depth, mean(y), flag);

        if(n>=min_node)            
            [best_attribute,best_threshold,std_dev,subleft,subright]=buildnode(X,y);
            if (std_dev<sigma)
                 tree.class = 1; 
                 tree.num = size(X);
                return;
            end
            y_left=y(subleft); 
            y_right=y(subright);
            X_left=X(subleft,:); 
            X_right=X(subright,:);
            fprintf('Column = %d. SplitValue = %f. StandardDeviation = %f.\n', best_attribute, best_threshold, std_dev);
           
            
            tree.kids = cell(1,2);
            depth = depth+1;
            tree.kids{1} = DecisionTreeLearning(X_left, y_left, depth, 1); 
            tree.kids{2} = DecisionTreeLearning(X_right, y_right, depth, 0);
            depth = depth-1;
            
            tree.op = [best_attribute,best_threshold];

        end        
    end

    function [best_attribute,best_threshold,std_dev,subleft,subright] = buildnode(X,y)
        [x_row, x_col] = size(X);
        std_dev = std(y);
        best_attribute = 0;
        for i = 1:x_col
            [best_threshold_i,std_dev_i,subleft_i,subright_i]=split(X(:,i),y);
            if(std_dev_i<std_dev)
                std_dev = std_dev_i;
                best_attribute = i;
            end
        end
        [best_threshold,std_dev,subleft,subright]=split(X(:,best_attribute),y);
    end
end


