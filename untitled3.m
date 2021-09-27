%% 图像取点操作
% 读入图片
depth_path='sample/data/depth/001_Kinect_FE_1DEPTH/01.png';
y=imread(depth_path);
imshow(y)%显示该图 
set(gcf,'outerposition',get(0,'screensize'));%使该图显示最大化，便于取点  
%利用鼠标取点，按回车键结束。这个时候应顺序点取图中坐标轴 左下，左上，右上，右下四个点。
[x0,y0] = ginput;
%开始撷取其中一条实线上的点，按你需要的精度，点取任意多的点，回车结束  %因为位图的屏幕坐标是从左上角为坐标原点开始的，需要做些变换： 
[x1,y1] = ginput; 
x1=(x1-min(x0))*25/(max(x0)-min(x0));%如果坐标原点不为0，则需在该轴加上 省的坐标轴数（此处为x轴加上375）；350是图像轴的标长；  
y1=(y1-max(y1))*4*10^4/(min(y0)-max(y0)); %y轴做同样的处理；
figure(2)
plot(x1,y1,'k.','Markersize',5); %画图，设置为黑色的点图  
axis([0,0.007,0,70]);%定义坐标范围  
set(gcf,'outerposition',get(0,'screensize'));%使该图显示最大化，便于观察




