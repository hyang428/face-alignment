fig = figure;

for i =1:length(Data)
    hold off;
    img = Data{i,1}.img_gray;
    imshow(img);
    hold on;
    rectangle('Position', Data{i,1}.bbox_gt);
    %{
    for j = 1:8
        pts = Data{i,1}.intermediate_shapes{1,j};
        plot(pts(:,1),pts(:,2),'*', 'color',rand(1,3));
        disp('Stage 1...');
        pause;
    end
    %}
    for j = 1:12
        pts = Data{i,1}.intermediate_shapes{1,8}(:,:,j);
        plot(pts(:,1),pts(:,2),'*','color',rand(1,3));
        disp('Different initial...');
        pause
    end
    
    disp('Press any key to continue...');
    pause;
end