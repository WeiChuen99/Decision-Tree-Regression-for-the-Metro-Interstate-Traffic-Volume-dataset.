% 01: calculate the standard deviation of target
% 02: split the dataset into different features, then calculate the standard deviation for each
% StandardDeviationReduction(T,X)=S(T)-S(T,X)
% 03: Choose feature with largest standard deviation reduction as decision node
% 04: Divide dataset based on the values of the selected feature
% 05: Branch set with standard deviation > 0 needs further splitting
% 06: Run recursively on the non-leaf branches
    
% Main starts here
clear
clc
data=readtable('Metro_Interstate_Traffic_Volume.csv');
attribute_name = data.Properties.VariableNames;

% Data processing 
holiday	=data(:,1);
holiday = table2array(holiday);
holiday = categorical(holiday);
holiday = array2table(double(holiday));
holiday.Properties.VariableNames = {'holiday'};

temp	=data(:,2);
rain_1h	=data(:,3);
snow_1h	=data(:,4);
clouds_all	=data(:,5);

weather_main=data(:,6);
weather_main = table2array(weather_main);
weather_main = categorical(weather_main);
weather_main = array2table(double(weather_main));
weather_main.Properties.VariableNames = {'weather_main'};

weather_description	=data(:,7);
weather_description = table2array(weather_description);
weather_description = categorical(weather_description);
weather_description = array2table(double(weather_description));
weather_description.Properties.VariableNames = {'weather_description'};

date_time	=data(:,8);
date_time = table2array(date_time);
date_time = datenum(date_time);
date_time = array2table(double(date_time));
date_time.Properties.VariableNames = {'date_time'};

X=[holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time];
X = table2array(X);

traffic_volume=data(:,end);
traffic_volume=table2array(traffic_volume);
y=traffic_volume;    

dataset = [X, y];

all_RMSE = 0;

%% Data preparation for cross-validation
k=10; % k-fold
sampleSize = 4820; % size of each set divisible by 10
lastn = 4; 
table = dataset(1:end-lastn,:); %remove last 2 rows
[r, ~] = size(dataset);
numTestItems = round(r*0.1); %size of test set
numTrainingItems = r - numTestItems; % leftover to be training set
dataIndices = randperm(r); % shuffle the dataset 
shuffled_data = dataset(dataIndices,:);

%% K-Fold cross validation
for fold =1:k
    fprintf(" %d Fold\n",fold);
    fprintf('-------------\n');
    test_indices = 1+(fold-1)*sampleSize:fold*sampleSize;
    train_indices = [1:(fold-1)*sampleSize, fold*sampleSize+1:numTrainingItems];
    
    %% Training data preparation
    trainingData = dataset(train_indices,:);
    testData = dataset(test_indices,:);
    Xtrain = trainingData(:,1:8);
    ytrain = trainingData(:,end);
    Xtest = testData(:,1:8);
    ytest = testData(:,end);
    
    % Build Regression Tree
    trees = DecisionTreeLearning(Xtrain,ytrain,1,1,attribute_name);
    DrawDecisionTree(trees);

    % Predict on test set base of regression tree
    prediction = predict(trees, Xtest);
    
    % Calculate RMSE for current fold
    RMSE = sqrt(mean((ytest-prediction).^2));

    % Accumulate all RMSE for every fold
    all_RMSE = all_RMSE + RMSE;   
end
%   Calculate average RMSE for all 10 folds
Average_RMSE = all_RMSE/k;

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

function  [tree]= DecisionTreeLearning(X,traffic_volume,depth,flag,attribute_name)    
    % Create tree node
    tree = struct('op','','kids',[],'class',[],'attribute',0,'threshold', 0);

    n = length(traffic_volume);
    min_node = 4338; % 10% of the number of observations
    sigma =0.0001;   % ?10?^(-5)

    fprintf('DepthofNode = %d. NodeValue = %1.f. FlagSign = %d.\n', depth, mean(traffic_volume), flag);
    tree.class = mean(traffic_volume);
    
    if(n>=min_node)            
        [best_attribute,best_threshold,std_dev,subleft,subright]=buildnode(X,traffic_volume);
        if (std_dev<sigma)
            tree.op = '';
            tree.attribute = 0;
            tree.threshold = 0;
            tree.kids = [];
            tree.class = mean(traffic_volume);
            return;
        end
        tree.op = char(attribute_name{best_attribute});
        tree.attribute = best_attribute;
        tree.threshold = best_threshold;

        y_left=traffic_volume(subleft); 
        y_right=traffic_volume(subright);
        X_left=X(subleft,:); 
        X_right=X(subright,:);

        fprintf('Column = %d. SplitValue = %1.f. StandardDeviation = %f.\n', best_attribute, best_threshold, std_dev);

        tree.kids = cell(1,2);
        depth = depth+1;

        tree.kids{1} = DecisionTreeLearning(X_left, y_left, depth, 1, attribute_name); 
        tree.kids{2} = DecisionTreeLearning(X_right, y_right, depth, 0, attribute_name);
        depth = depth-1; 
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

function [prediction_result] = predict(tree, examples)
    root = tree;
    len_examples = size(examples, 1);
    prediction_result = transpose(1:len_examples);
    % While we still have subtree, iterate through the subtrees
    for row = 1:len_examples
        tree = root;
         while tree.op ~= -1
            if examples(row, tree.attribute) <= tree.threshold
                tree = tree.kids{1}; 
            else
                tree = tree.kids{2};
            end
         end
         prediction_result(row,1) = tree.class;
    end    
end
