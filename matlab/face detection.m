I=imread('xxx.jpg');
figure;
subplot(231);
imshow(I);
title('initial one');

I=rgb2gray(I);
I=wiener2(I,[5 5]);
subplot(232);
imshow(I);
title('gray one');

BW=im2bw(I);
subplot(233);
imshow(BW);
title('binary one');

[n1 n2]=size(BW);
r=floor(n1/10);
c=floor(n2/10);
x1=1;
x2=r;
s=r*c;
for i=1:10
    y1=1;
    y2=c;
    for j=1:10
        if(y2<=c|y2>=9*c)|(x1==1|x2==r*10)
            BW(x1:x2,y1:y2)=0
        end
        y1=y1+c;
        y2=y2+c;
    end
    x1=x1+r;
    x2=x2+r;
end
subplot(234);
imshow(BW);
title('no background one');

L=bwlabel(BW,4);
B1=regionprops(L,'Boundingbox');
B2=struct2cell(B1);
B3=cell2mat(B2);
[s1 s2]=size(B3);
mx=0;
for k=3:4:s2-1
    p=B3(1,k)*B3(1,k+1);
    if p>mx&(B3(1,k+1)/B3(1,k))<2
        mx=p;
        j=k;
    end
end
subplot(235);
imshow(I);
title('facial detection');

hold on;
rectangle('Position',[B3(1,j-2),B3(1,j-1),B3(1,j),B3(1,j+1)],'EdgeColor','r');
