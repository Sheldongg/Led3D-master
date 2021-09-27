function [depth,mask] = calcDepthAndNormal(vertex,scale,Use_preprocess)
%     if size(vertex,1)==3
%         pc=vertex';
%     else
%         pc=vertex;
%     end
    pc=vertex'; %这一撇是转置的意思，xyzpoints的输入必须是Mx3
    
    pc(:,3)=max(pc(:,3))-pc(:,3); %矩阵的底三列元素
    
%     point_cloud=pointCloud(pc);
    %point_cloud=pcdenoise(point_cloud); %点云去躁
%     vertex=point_cloud.Location()';
    
%要旋转的矩阵dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale)
%角度 -90
%方法：双线性
%      depth=imrotate(dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale),-90,'bilinear');
%     depth=dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale));
    depth=dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale);

    mask=depth;

    if Use_preprocess    

    end   

    nn=size(mask,1);
    mm=size(mask,2);

max_x=nn;
min_y=1;
min_x=1;
max_y=mm;   
    croped_face=depth(min_x:max_x,min_y:max_y);
    depth=croped_face;
    croped_mask=mask(min_x:max_x,min_y:max_y);
    mask=croped_mask;    
end