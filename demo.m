
%addpath('util');
%加载坐标，2x眼睛、鼻子、2x嘴巴
for n =1:1
    x = 10*n;
    %x=10;
    str1=mat2str(x);
    str2='.txt';
    path2='.png';
    path3='.jpg';
    name='lidong0.5/';
    landmark_path_1='sample/label/infrared/lidong0.5/';
    landmark_path=[landmark_path_1,str1,str2];
    %color_path='sample/data/color/001_Kinect_FE_1COLOR/01.jpg';
%     depth_path_='sample/data/depth/001_Kinect_FE_1DEPTH/lidong0.5/';
    depth_path='/home/alien/save/build/depth/13.001866.png'
%     depth_path=[depth_path_,str1,path2];
    %color_face = imread(color_path);    figure(2),imshow(color_face);  
    depth_face = imread(depth_path);    figure(3),imshow(depth_face,[500,800]);    

    landmark = readLandmark(landmark_path);
    %nose_tip = calcNTP(landmark);
    nose_tip = landmark(:,3); %第三列

    %extract roi by nose tip
    %根据手动标记的鼻尖的 x 和 y 坐标，从原始深度帧（512 × 424）中裁剪出 180 × 180 的脸，并将其线性插值到 360 × 360
    imgSize = [180, 180];
    roi=depth_face(nose_tip(2)-imgSize(1)/2 +1:nose_tip(2)+imgSize(1)/2 ,nose_tip(1)-imgSize(2)/2 +1:nose_tip(1)+imgSize(2)/2);
    figure(4),imshow(roi,[500,800]);
    
 
    
    reSize=360;
    roi_face = imresize(roi,[reSize,reSize],'bilinear','AntiAliasing',false);

    %x(i,j,k)的含义是第k层矩阵的第i行第j列元素;
    %floor 朝负无穷大方向取整
    %mod 取模 点云图,这部分应该是取出异常值  
    pc_template=zeros(3,reSize*reSize);
    pc_template(1,:) = floor((0:(reSize*reSize-1)) /reSize)+1;
    pc_template(2,:) = mod(0:(reSize*reSize-1),reSize)+1;
    pc_template(3,:) = roi_face(:);%将所有roi_face数据变成pc_template的第三行
    figure(5),pcshow(pointCloud(pc_template'));


    r=100;
    %基于x和y坐标在给定鼻尖周围定位一个10×10的面片，并使用其中值而不是平均值作为修改点
    xo=double(reSize/2); yo=double(reSize/2);  
    zo=double(median(median(roi_face(xo-10:xo+10,yo-10:yo+10))));
%     zo=493;
    pc_template(3,((xo-pc_template(2,:)).*(xo-pc_template(2,:))+(yo-pc_template(1,:)).*(yo-pc_template(1,:))+(zo-pc_template(3,:)).*(zo-pc_template(3,:)))>r*r)=0;
    pc_face=pc_template(:,pc_template(3,:)>0);
    figure(6),pcshow(pointCloud(pc_face'));
    
    
    
%     [depth,mask] = calcDepthAndNormal(pc_face,1,0);
    pc=pc_face'; %这一撇是转置的意思，xyzpoints的输入必须是Mx3
    
    pc(:,3)=max(pc(:,3))-pc(:,3); %矩阵的底三列元素
    
%     point_cloud=pointCloud(pc);
    %point_cloud=pcdenoise(point_cloud); %点云去躁
%     vertex=point_cloud.Location()';
    
%要旋转的矩阵dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale)
%角度 -90
%方法：双线性
%      depth=imrotate(dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale),-90,'bilinear');
%     depth=dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))*scale),uint8((max(pc(:,2))-min(pc(:,2))))*scale));
%     depth=dep(pc,uint8((max(pc(:,1))-min(pc(:,1)))),uint8((max(pc(:,2))-min(pc(:,2)))));
% function ret = dep(data, size_x, size_y)
    size_x = uint8((max(pc(:,1))-min(pc(:,1))))
    size_y = uint8((max(pc(:,2))-min(pc(:,2))))
    x = pc(:,1);
    y = pc(:,2);
    z = pc(:,3);
   
    max_x = max(x);
    min_x = min(x);
    max_y = max(y);
    min_y = min(y);
    min_z = min(z);
    max_z = max(z);
    range_x = max_x - min_x;
    range_y = max_y - min_y;
    ret = zeros(size_x, size_y);
    ret(:,:) = min_z;
    len = size(x, 1);
%     min_z = 999;
    for i = 1:len
        X = floor((x(i)- min_x) / range_x * (size_x-1))+1;
        Y = floor((y(i)- min_y) / range_y * (size_y-1))+1;
        ret(X, Y) = max(ret(X, Y), z(i));
        min_z = min(min_z, ret(X, Y));
    end
    range_z = max_z - min_z;
    ret = max(ret, min_z);
    
    depth = round((ret - min_z) / range_z * 255);
% end

    mask=depth;
  

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
    
    
    m=uint8(depth)
    figure(77),imshow(uint8(depth))
    %figure(78),imshow(uint8(mask))
    depth=normalizeValue(depth);
    figure(88),imshow(uint8(depth));
    
    
    depth=normalizeSize(depth);
    figure(99),imshow(uint8(depth));
    
    %DE
    img_size =180;
    figure(7),imshow(uint8(depth));
    normal = calcNormal(uint8(depth));
    figure(8),imshow(normal);%得到normal图
    depth=imresize(depth,[img_size img_size]);
    normal=imresize(normal,[img_size img_size]);

    path1='./result/test/';
    path2='./result/test2/';
    path1_=[path1,name,str1,path3];
    path2_=[path2,name,str1,path3];
    imwrite(uint8(depth),path1_);
    %%imwrite(uint8(depth),'./python/data/lidong/gallery/lidong/depth.jpg');%这个就是这个demo识别时候用到的图了

    imwrite(uint8(normal),path2_);%

%     pc_face=pcdenoise(pointCloud(pc_face'));
%     selected_pc=pc_face.Location';%max(selected_pc(3,:)),min(selected_pc(3,:))
%     fid = fopen('result/face.obj','wt');
%     fprintf(fid, 'v %d %d %.2f\n',selected_pc);
%     fclose(fid);


%     %------------姿势生成
%     face_vertex=readOBJ('result/face.obj');
%     type='PoseVariation';
%     degree=20;degree_1=40;degree_2=60;
%     rotation=[[0,0,0];
%               [0,degree/180*pi,0];[0,-degree/180*pi,0];[degree/180*pi,0,0];[-degree/180*pi,0,0];
%               [0,degree_1/180*pi,0];[0,-degree_1/180*pi,0];
%               [0,degree_2/180*pi,0];[0,-degree_2/180*pi,0]];
%     rotation_type={'nu',...
%         'yaw=20','yaw=-20','pitch=20','pitch=-20',...
%         'yaw=40','yaw=-40',...
%         'yaw=60','yaw=-60',};
%     img_size=128;
% 
%     for r_i =1:size(rotation,1)
%         vertex=(face_vertex'*RotationMatrix(rotation(r_i,1),rotation(r_i,2),rotation(r_i,3)))';
%         [depth,mask]=calcDepthAndNormal(vertex,1,1);
%         if size(depth,1)<50
%             continue;
%         end
%         depth=normalizeValue(depth);%imshow(uint8(depth));
%         depth=normalizeSize(depth);mask=normalizeSize(mask);
%         normal=calcNormal(depth);%imshow(uint8(normal));
% 
%         depth=imresize(depth,[img_size img_size]);
%         normal=imresize(normal,[img_size img_size]);
%         imwrite(uint8(depth),['result/' rotation_type{r_i} '.jpg']);
%         imwrite(uint8(normal),['result/' rotation_type{r_i} '.jpg']);
%     end
%     %--------------------------------------------------------------------------------
%     type='ShapeVariation';
%     [depth,mask]=calcDepthAndNormal(pc_face.Location',1,1);
%     %     Shape Jittering 形状抖动
%     [ depth_noise,normal_noise ] = noise( depth,mask,'gaussian',0,0.00002 );
%     depth_noise=imresize(depth_noise,[img_size img_size]);
%     normal_noise=imresize(normal_noise,[img_size img_size]);
%     imwrite(uint8(depth_noise),'result/depth_Jittering.jpg');
%     imwrite(uint8(normal_noise),'result/normal_Jittering.jpg'); 
%     %     Shape Scaling 形状缩放
%     scale=1.1;
%     [depth_shrink,normal_shrink]=shrink(depth,mask,scale);
%     depth_shrink=imresize(depth_shrink,[img_size img_size]);
%     normal_shrink=imresize(normal_shrink,[img_size img_size]);
%     imwrite(uint8(depth_shrink),'result/depth_Scaling.jpg');
%     imwrite(uint8(normal_shrink),'result/normal_Scaling.jpg');
end
