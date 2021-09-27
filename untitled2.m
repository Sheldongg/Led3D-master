% 单层切片遍历所有点，标出鼻尖置信点和两个交点
clc;
read_file = importdata('/home/alien/Downloads/lidong/Led3D-master/sample/data/depth/001_Kinect_FE_1DEPTH/01.png'); %读取文件
file_x = read_file(:,1);
file_y = read_file(:,2);
file_z = read_file(:,3);
% set(figure,'name','点云');
% plot3(file_x,file_y,file_z,'.')
[X,Y,Z] = griddata(file_x,file_y,file_z,linspace(min(file_x),max(file_x),200)',linspace(min(file_y),max(file_y),20),'cubic'); %插值
% set(figure,'name','网格');
% surf(Z);  %显示三维曲面
 
xz = Z(13,:); %获取水平切片,这里的13是自己随便取的一层切片
x = 1:200;
yqu = xz(1,x);
set(figure,'name','切片');
plot(x,yqu);
%画圆
I = find(~isnan(xz)); %获取水平切片非空位置
[x_s,~] = min(I); %水平切片有效起始位置
[x_e,~] = max(I); %水平切片有效终止位置
%r = round((x_e-x_s)/5); %自适应因切圆半径(效果并不好)
r = 30; %指定切圆的半斤，效果更好
h_gaos = []; %切片中每个点作为圆心和切片形成的交点所形成的三角形的高集合
for x1 = x_s+r:x_e-r %圆心的可取值范围
    if x1>1 %避免水平切片太小
        z1 = xz(1,x1); %切片中对应z的坐标
        i = x1-r:x1+r; %圆心的取值范围
        y2 = sqrt(r.^2-(i-x1).^2)+z1; %切圆的z坐标（上半圆）
        y = xz(1,i);
        cy = y2-y;
        pos = cy>0;
        neg = cy<=0;
        %确定变号位置
        fro = diff(pos)~=0; %变号的前导位置
        rel = diff(neg)~=0; %变号的尾巴位置
        zpf = find(fro==1); %记录索引
        zpr = find(rel==1)+1; %记录索引
        zpfr = [zpf,zpr];
        x0 = (i(zpr).*(y2(zpf)-y(zpf))-i(zpf).*(y2(zpr)-y(zpr)))./(y(zpr)+y2(zpf)-y(zpf)-y2(zpr));
        y0 = y(zpf)+(x0-i(zpf)).*(y(zpr)-y(zpf))./(i(zpr)-i(zpr)-i(zpf));
        x0 = [x0 x0].';
        y0 = [y0 y0].';
        jie = unique([x0,y0],'rows'); %得到切片与下半圆的交点集合
        [mm,nn] = size(jie);
        %fprintf('解的个数：');
        %disp(mm);
        if mm == 2;
            jie1 = jie(1,:);
            jie2 = jie(2,:);
            if jie1(:,1)<x1 && jie2(:,1)>x1 %确保两个交点在圆心的两侧
                l1 = ((jie1(:,1)-x1)^2+(jie1(:,2)-z1)^2)^(1/2);
                l2 = ((jie2(:,1)-x1)^2+(jie2(:,2)-z1)^2)^(1/2);
                l3 = ((jie1(:,1)-jie2(:,1))^2+(jie1(:,2)-jie2(:,2))^2)^(1/2); %内接三角形的底为
                p = (l1+l2+l3)/2;
                s = sqrt(p*(p-l1)*(p-l2)*(p-l3)); %内接三角形的面积
                h_gao = 2*s/l3;  %内接三角形的高为
                h_gaos = [h_gaos;[x1,h_gao]]; %形成的高的集合
            end
        end
    end
end
if length(h_gaos)>0
    [m,p] = max(h_gaos(:,2));
    x_gao = h_gaos(p,1); %得到最大的高的圆心x坐标
    % x_gaos = [x_gaos;[j_new,x_gao,m]]; %切片Y坐标，圆心X坐标，最大三角形的高
end
disp('鼻尖置信值：');
disp(h_gaos);
if length(h_gaos)>0
    disp(size(h_gaos));
    [m,p]=max(h_gaos(:,2));
    x_gaos=h_gaos(p,1);
    z_gaos=xz(1,x_gaos);
    fprintf('最大三角形的x坐标为：');
    disp(x_gaos);
    fprintf('最大三角形的高为：');
    disp(m);
    hold on;
    i = x_gao-r:x_gao+r; %圆心的取值范围
    y2 = sqrt(r.^2-(i-x_gao).^2)+z_gaos; %切圆的z坐标（上半圆）
    y = xz(1,i);
    h=plot(i,y2,'b');
    cy = y2-y;
    pos = cy>0;
    neg = cy<=0;
    %确定变号位置
    fro = diff(pos)~=0; %变号的前导位置
    rel = diff(neg)~=0; %变号的尾巴位置
    zpf = find(fro==1); %记录索引
    zpr = find(rel==1)+1; %记录索引
    zpfr = [zpf,zpr];
    hold on;
    % 线性求交
    x0 = (i(zpr).*(y2(zpf)-y(zpf))-i(zpf).*(y2(zpr)-y(zpr)))./(y(zpr)+y2(zpf)-y(zpf)-y2(zpr));
    y0 = y(zpf)+(x0-i(zpf)).*(y(zpr)-y(zpf))./(i(zpr)-i(zpr)-i(zpf));
    x0 = [x0 x0].';
    y0 = [y0 y0].';
    hc=plot(x0,y0,'r*',x_gao,z_gaos,'g*');
    legend([h;hc],'切平面','交点','圆心','Location','northwest');
    xlabel('x'),ylabel('y'),zlabel('z');
    title('平面曲线焦点')
else
    disp('找不到置信点');
end