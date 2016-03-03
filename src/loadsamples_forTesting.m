function Data = loadsamples_forTesting(imgpathlist, exc_setlabel)
%LOADSAMPLES Summary of this function goes here
%   Function: load samples from dbname database
%   Detailed explanation goes here
%   Input: 
%        dbname: the name of one database
%        exc_setlabel: excluded set label
%   Output:
%        Data: loaded data from the database
%imgpathlist = textread(imgpathlistfile, '%s', 'delimiter', '\n');

Data = cell(length(imgpathlist), 1);

setnames = {'train' 'test'};

% However, the problem here is that I don't have the Computer Vision
% toolbox
% Create a cascade detector object.
% faceDetector = vision.CascadeObjectDetector();
% bboxes_facedet = zeros(length(imgpathlist), 4);
% bboxes_gt      = zeros(length(imgpathlist), 4);
% isdetected = zeros(length(imgpathlist), 1);

%parfor i = 1:length(imgpathlist)
for i = 1:length(imgpathlist)
    % read the image and obtain the width and height
    img = im2uint8(imread(imgpathlist{i}));
    Data{i}.width_orig    = size(img, 2);
    Data{i}.height_orig   = size(img, 1);
    
    % Data{i}.img      = img
    
    % This may have problem, casue we don't know the ground-truth for
    % testing.
  %  shapepath = strcat(imgpathlist{i}(1:end-3), 'pts');   % shape path
  %  Data{i}.shape_gt = double(loadshape(shapepath));      % load the shape ground-truth
    % Data{i}.shape_gt = Data{i}.shape_gt(params.ind_usedpts, :);
    % bbox     =  bounding_boxes_allsamples{i}.bb_detector; %
  %  Data{i}.bbox_gt = getbbox(Data{i}.shape_gt); % [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)];
  %  Data{i}.bbox_facedet = Data{i}.bbox_gt;
    % from now on, the shape_gt is unknown
    Data{i}.shape_gt = [];
 %{   
    % load the box
    boxpath = [imgpathlist{i} '.box'];
    box = loadbox(boxpath);
    Data{i}.bbox_gt = box;
    Data{i}.bbox_facedet = box;
 %}
    
    imshow(img);
    [x y] = ginput(2);
    
    gcf;
    Data{i}.bbox_gt = [x(1), y(1), x(2)-x(1), y(2)-y(1)];
    Data{i}.bbox_facedet = Data{i}.bbox_gt;
    % cut original image to a region which is a bit larger than the face
    % bounding box
   % region = enlargingbbox(Data{i}.bbox_gt, 2.0);
    region = enlargingbbox(Data{i}.bbox_gt, 1.2);
    
    region(2) = double(max(region(2), 1));
    region(1) = double(max(region(1), 1));
    
    bottom_y = double(min(region(2) + region(4) - 1, Data{i}.height_orig));
    right_x = double(min(region(1) + region(3) - 1, Data{i}.width_orig));
    
    img_region = img(region(2):bottom_y, region(1):right_x, :);
    
    Data{i}.bbox_gt(1) = Data{i}.bbox_gt(1) - double(region(1));
    Data{i}.bbox_gt(2) = Data{i}.bbox_gt(2) - double(region(2));
    Data{i}.bbox_facedet = Data{i}.bbox_gt;
    
    % Data{i}.shape_gt = bsxfun(@minus, Data{i}.shape_gt, double([region(1) region(2)]));
    
    % perform face detection using matlab face detector
    %{
    bbox = step(faceDetector, img_region);
    if isempty(bbox)
        % if face detection is failed        
        isdetected(i) = 1;
        Data{i}.bbox_facedet = getbbox(Data{i}.shape_gt);
    else
        int_ratios = zeros(1, size(bbox, 1));
        for b = 1:size(bbox, 1)
            area = rectint(Data{i}.bbox_gt, bbox(b, :));
            int_ratios(b) = (area)/(bbox(b, 3)*bbox(b, 4) + Data{i}.bbox_gt(3)*Data{i}.bbox_gt(4) - area);            
        end
        [max_ratio, max_ind] = max(int_ratios);
        
        if max_ratio < 0.4  % detection fail
            isdetected(i) = 0;
        else
            Data{i}.bbox_facedet = bbox(max_ind, 1:4);
            isdetected(i) = 1;
            % imgOut = insertObjectAnnotation(img_region,'rectangle',Data{i}.bbox_facedet,'Face');
            % imshow(imgOut);
        end   
    end
    %}
    % recalculate the location of groundtruth shape and bounding box
    % Data{i}.shape_gt = bsxfun(@minus, Data{i}.shape_gt, double([region(1) region(2)]));
    % Data{i}.bbox_gt = getbbox(Data{i}.shape_gt);
    
    if size(img_region, 3) == 1
        Data{i}.img_gray = img_region;
    else
        % hsv = rgb2hsv(img_region);
        Data{i}.img_gray = rgb2gray(img_region);
    end    
    
    Data{i}.width    = size(img_region, 2);
    Data{i}.height   = size(img_region, 1);
end

ind_valid = ones(1, length(imgpathlist));
%parfor i = 1:length(imgpathlist)
for i = 1:length(imgpathlist)
    if ~isempty(exc_setlabel)
        ind = strfind(imgpathlist{i}, setnames{exc_setlabel});
        if ~isempty(ind) % | ~isdetected(i)
            ind_valid(i) = 0;
        end
    end
end

% learn the linear transformation from detected bboxes to groundtruth bboxes
% bboxes = [bboxes_gt bboxes_facedet];
% bboxes = bboxes(ind_valid == 1, :);

Data = Data(ind_valid == 1);

end

%% read the box
function box = loadbox(path)
file = fopen(path);
box = textscan(file, '%f %f %f %f');
box = cell2mat(box);
fclose(file);

end

%% read the landmarks from the points files
function shape = loadshape(path)
% function: load shape from pts file
file = fopen(path);

if ~isempty(strfind(path, 'COFW'))
    shape = textscan(file, '%d16 %d16 %d8', 'HeaderLines', 3, 'CollectOutput', 3);
else
    shape = textscan(file, '%d16 %d16', 'HeaderLines', 3, 'CollectOutput', 2);
end
fclose(file);

shape = shape{1};
end

%% scale the bbox, and make it a little bigger than the detected box
%  why we need to scale it?
%     cause the detected box is not 100% accurate, so, part of the face may
%     be not included in the box, which will cause errors for the following
%     procedures.
function region = enlargingbbox(bbox, scale)

region(1) = floor(bbox(1) - (scale - 1)/2*bbox(3));
region(2) = floor(bbox(2) - (scale - 1)/2*bbox(4));

region(3) = floor(scale*bbox(3));
region(4) = floor(scale*bbox(4));

% region.right_x = floor(region.left_x + region.width - 1);
% region.bottom_y = floor(region.top_y + region.height - 1);


end

