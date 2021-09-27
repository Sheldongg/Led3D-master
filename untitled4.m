clear all
close all
clc
depth_path='sample/data/depth/001_Kinect_FE_1DEPTH/10.png';
set(gcf,'color','white')
A=imread(depth_path);
B=imshow(A);
x=ginput;
k=size(x,1);
axis on
hold on
for i=1:k
    plot(x(i,1),x(i,2),'r+')
    text(x(i,1),x(i,2),sprintf('(%f,%f)',x(i,1),x(i,2)),'Color','red')
end  
axis off