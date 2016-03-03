%TRAIN_MODEL Summary of this function goes here
%   Function: test face alignment model
%   Detailed explanation goes here
%   Input:
%       dbnames: the names of database
% Configure the parameters for training model
global params;
config_te;

dbnames = {'data'};
dataType ={'*.png'};    % the corresponding type for the dataset
DataPath = './';

load('./initial_shape/InitialShape_68.mat');
params.meanshape        = S0;

% load trainning data from hardware
Te_Data = [];
for i = 1:length(dbnames)
    % load training samples (including training images, and groundtruth shapes)
    filelist = dir([DataPath, dbnames{i}, '/', dataType{i}]);
    for j =1:length(filelist)
        imgpathlist{j} = [DataPath, dbnames{i}, '/',filelist(j).name];
    end 
    % imgpathlist
    te_data = loadsamples_forTesting(imgpathlist, 1);
    Te_Data = [Te_Data; te_data];
end


% Augmentate data for traing: assign multiple initial shapes to each image

Data = Te_Data;
Param = params;

% choose corresponding points for training
for i = 1:length(Data)
    % modify detection boxes 
    shape_facedet = resetshape(Data{i}.bbox_facedet, Param.meanshape);
    shape_facedet = shape_facedet(Param.ind_usedpts, :);
    Data{i}.bbox_facedet = getbbox(shape_facedet);
    
end

Param.meanshape        = S0(Param.ind_usedpts, :);
dbsize = length(Data);

augnumber = Param.augnumber;

for i = 1:dbsize        
    % initializ the shape of current face image by randomly selecting multiple shapes from other face images       
    % indice = ceil(dbsize*rand(1, augnumber));  

    indice_rotate = ceil(dbsize*rand(1, augnumber));  
    indice_shift  = ceil(dbsize*rand(1, augnumber));  
    scales        = 1 + 0.2*(rand([1 augnumber]) - 0.5);
    
    Data{i}.intermediate_shapes = cell(1, Param.max_numstage);
    Data{i}.intermediate_bboxes = cell(1, Param.max_numstage);
    
    Data{i}.intermediate_shapes{1} = zeros([size(Param.meanshape), augnumber]);
    Data{i}.intermediate_bboxes{1} = zeros([augnumber, size(Data{i}.bbox_gt, 2)]);
    
    Data{i}.shapes_residual = zeros([size(Param.meanshape), augnumber]);
    Data{i}.tf2meanshape = cell(augnumber, 1);
    Data{i}.meanshape2tf = cell(augnumber, 1);
        
    % if Data{i}.isdet == 1
    %    Data{i}.bbox_facedet = Data{i}.bbox_facedet*ts_bbox;
    % end     
    for sr = 1:params.augnumber
        if sr == 1
            % estimate the similarity transformation from initial shape to mean shape
            % Data{i}.intermediate_shapes{1}(:,:, sr) = resetshape(Data{i}.bbox_gt, Param.meanshape);
            % Data{i}.intermediate_bboxes{1}(sr, :) = Data{i}.bbox_gt;
            Data{i}.intermediate_shapes{1}(:,:, sr) = resetshape(Data{i}.bbox_facedet, Param.meanshape);
            Data{i}.intermediate_bboxes{1}(sr, :) = Data{i}.bbox_facedet;
            
            meanshape_resize = resetshape(Data{i}.intermediate_bboxes{1}(sr, :), Param.meanshape);
                        
            Data{i}.tf2meanshape{1} = fitgeotrans(bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, 1), mean(Data{i}.intermediate_shapes{1}(1:end,:, 1))), ...
                (bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :)))), 'NonreflectiveSimilarity');
            Data{i}.meanshape2tf{1} = fitgeotrans((bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :)))), ...
                bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, 1), mean(Data{i}.intermediate_shapes{1}(1:end,:, 1))), 'NonreflectiveSimilarity');
                        
        else
            % randomly rotate the shape            
            % shape = resetshape(Data{i}.bbox_gt, Param.meanshape);       % Data{indice_rotate(sr)}.shape_gt
            shape = resetshape(Data{i}.bbox_facedet, Param.meanshape);       % Data{indice_rotate(sr)}.shape_gt
            
            if params.augnumber_scale ~= 0
                shape = scaleshape(shape, scales(sr));
            end
            
            if params.augnumber_rotate ~= 0
                shape = rotateshape(shape);
            end
            
            if params.augnumber_shift ~= 0
                shape = translateshape(shape, Data{indice_shift(sr)}.shape_gt);
            end
            
            Data{i}.intermediate_shapes{1}(:, :, sr) = shape;
            Data{i}.intermediate_bboxes{1}(sr, :) = getbbox(shape);
            
            meanshape_resize = resetshape(Data{i}.intermediate_bboxes{1}(sr, :), Param.meanshape);
                        
            Data{i}.tf2meanshape{sr} = fitgeotrans(bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, sr), mean(Data{i}.intermediate_shapes{1}(1:end,:, sr))), ...
                bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), 'NonreflectiveSimilarity');
            Data{i}.meanshape2tf{sr} = fitgeotrans(bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), ...
                bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, sr), mean(Data{i}.intermediate_shapes{1}(1:end,:, sr))), 'NonreflectiveSimilarity');
                        
            % Data{i}.shapes_residual(:, :, sr) = tformfwd(Data{i}.tf2meanshape{sr}, shape_residual(:, 1), shape_residual(:, 2));
        end
    end
end

% test random forests
 %load('./models/LBFRegModel_afw_lfpw.mat');

randf = LBFRegModel.ranf;
Ws    = LBFRegModel.Ws;

dbname_str = '';
for i = 1:length(dbnames)
    dbname_str = strcat(dbname_str, dbnames{i}, '_');
end
dbname_str = dbname_str(1:end-1);

if ~exist(dbname_str,'dir')
   mkdir(dbname_str);
end
    
for s = 1:params.max_numstage
    % derive binary codes given learned random forest in current stage
    
    disp('extract local binary features...');
    % if ~exist(strcat(dbname_str, '\lbfeatures_', num2str(s), '.mat'))
        tic;
        binfeatures = derivebinaryfeat(randf{s}, Data, Param, s);
        % binfeatures = derivebinaryfeat(TrModel{s}.RF, Data, Param, min(s,  params.max_numstage));
        toc;    
        % save(strcat(dbname_str, '\lbfeatures_', num2str(min(s,  params.max_numstage)), '.mat'), 'binfeatures');
    % else
    %     load(strcat(dbname_str, '\lbfeatures_', num2str(s), '.mat'));
    % end
    % predict the locations of landmarks in current stage
    tic;
    disp('predict landmark locations...');

    Data = globalprediction(binfeatures, Ws{s}, Data, Param, s);        
    % Data = globalprediction(binfeatures, TrModel{s}.W, Data, Param, min(s,  params.max_numstage));        
    toc;        
    
end
