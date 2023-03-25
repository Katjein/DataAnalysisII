close all
clear all
clc
%% flags for plots
plot_original=true;
plot_redchannel=true;
plot_greenchannel=true;
plot_bluechannel=true;
plot_rawmask_original=true;
plot_rawmask_white=true;
plot_finemask=true;
plot_contour=true;
plot_histogram=true;
plot_red_hist=true;
plot_green_hist=true;
plot_blue_hist=true;
plot_rawpressure=true;
plot_finepressure=true;
plot_diffpressmasks=true;
plot_netpressmasks=true;
plot_netpressure=true;

find_toeline=false; % better leave this on "false", procedure not yet developed
net_classification=true;
nfig=0;

%% load image data
filename='PK_06_11_2012_li.jpg';
%filename='10-253-R.jpg';
%filename='10-258-L.jpg';

p=imread(filename);
if p(1,1,1)<100  %automatic detection of background color based on first pixel
    background=0;
else
    background=1;
end
pd=cast(p,'double')/255; % convert to double
ny=size(pd,1);
nx=size(pd,2);
%% add cw pixels on each side, necessary for contour
cw=2;   %contourwidth
if background==0
    plarge=zeros(ny+2*cw, nx+2*cw,3);
else 
    plarge=ones(ny+2*cw, nx+2*cw,3);
end
plarge(cw+1:ny+cw,cw+1:nx+cw,1:3)=pd(:,:,:);
pd=plarge;
ny=size(pd,1);
nx=size(pd,2);
pblack=zeros(ny,nx,3);
footmask=pblack;

%% plot image
if plot_original
    nfig=nfig+1;
    figure(nfig);
    image(pd);
    axis equal;
    title('Orginial');
end

%% plot channels
if plot_redchannel
    nfig=nfig+1;
    figure(nfig);
    plotmask=pblack;
    plotmask(:,:,1)=1;
    image(pd.*plotmask);
    axis equal
    title('Red Channel');
end
if plot_greenchannel
    nfig=nfig+1;
    figure(nfig);
    plotmask=pblack;
    plotmask(:,:,2)=1;
    image(pd.*plotmask);
    axis equal
    title('Green Channel');
end
if plot_bluechannel
    nfig=nfig+1;
    figure(nfig);
    plotmask=pblack;
    plotmask(:,:,3)=1;
    image(pd.*plotmask);
    axis equal
    title('Blue Channel');
end

%% footmask
footmask=pblack;

if background==1       % if background is white
    thresh=[0.95 0.95 0.95];
    for x=1:nx
        for y=1:ny
            if min([pd(y,x,1)-thresh(1) pd(y,x,2)-thresh(2) pd(y,x,3)-thresh(3)])<0
                footmask(y,x,:)=1;
            end
        end
    end
else                    % if background is black
    thresh=[0.05 0.05 0.05];
    for x=1:nx
        for y=1:ny
            if max([pd(y,x,1)-thresh(1) pd(y,x,2)-thresh(2) pd(y,x,3)-thresh(3)])>0
                footmask(y,x,:)=1;
            end
        end
    end   
end


if plot_rawmask_white
    nfig=nfig+1;
    figure(nfig);
    image(footmask);
    axis equal
    title('Foot Mask_White');
end


%% contour

contourmask=pblack;
footmask2=footmask;
clrdist=zeros(256,3);
intclrdist=clrdist;
footpixcount=0;

offratio=0.2;   %% ratio of off-foot-pixels in neigboorhood in order to change to background
ncrit=round((1-offratio)*(2*cw+1)^2);    %% how many pixels need to be on foot;

for y=cw+1:ny-cw
    for x=cw+1:nx-cw
        if background % apply correction only to white background
            nfootpixel=sum(sum(footmask(y-cw:y+cw, x-cw:x+cw,1))); %% how many pixel in array are on foot           
            if nfootpixel<=ncrit                   
                footmask2(y,x,1:3)=0;
            else
                footmask2(y,x,1:3)=1;
            end
        end
        % do counts to prepare histogramm
        if footmask2(y,x,1)
            footpixcount=footpixcount+1;
            clrdist(p(y-cw,x-cw,1)+1,1)=clrdist(p(y-cw,x-cw,1)+1,1)+1;
            clrdist(p(y-cw,x-cw,2)+1,2)=clrdist(p(y-cw,x-cw,2)+1,2)+1;
            clrdist(p(y-cw,x-cw,3)+1,3)=clrdist(p(y-cw,x-cw,3)+1,3)+1;
        end
    end
    contleft=find(footmask2(y,:,1),1,'first');
    contright=find(footmask2(y,:,1),1,'last');
    contourmask(y,contleft,1:3)=1;
    contourmask(y,contright,1:3)=1;
end

for x=1:nx     % add horizontal lines in contour
    conttop=find(footmask2(:,x,1),1,'first');
    contbottom=find(footmask2(:,x,1),1,'last');
    contourmask(conttop,x,1:3)=1;
    contourmask(contbottom,x,1:3)=1;  
end

intclrdist(1,:)=clrdist(1,:);
for idx=2:256
    intclrdist(idx,:)=intclrdist(idx-1,:)+clrdist(idx,:);  % integrate colour distribution
end

%% plot what we have so far
if plot_rawmask_original
    nfig=nfig+1;
    figure(nfig);
    image(footmask2);
    axis equal
    title('Foot Mask');
end

if plot_finemask
    nfig=nfig+1;
    figure(nfig);
    image(footmask2.*pd);
    axis equal
    title('Foot Mask corrected');
end

if plot_contour
    nfig=nfig+1;
    figure(nfig);
    image(contourmask);
    axis equal
    title('Contour');
end

%% find max_width

w=zeros(ny,1);
for y=1:ny
    onidx=find(footmask2(y,1:nx,1));
    if size(onidx,2)==0
        w(y)=0;
    else
        w(y)=max(onidx)-min(onidx);
    end
end    
[maxw maxwidx]=max(w);
hold on
lineleft=find(footmask2(maxwidx,1:nx),1,'first');
line([lineleft lineleft+w(maxwidx)],[maxwidx maxwidx],'Color','r');


%% find toeline  (just an example for further analyses, not perfect yet, needs to be improved)
if find_toeline
    intensity=zeros(1,maxwidx);
    vert_intensity_fit=zeros(nx,maxwidx);
    yvalid=vert_intensity_fit;
    xvalid=zeros(1,nx);
    orgyidx=vert_intensity_fit;
    xcount=0;
    yidx=zeros(1,nx);
    firstpix=zeros(1,nx);
    npix_valid=30;
   
    for x=1:nx
        pixcount=0;
        firstpix(x)=1;
        for y=1:maxwidx
            if footmask2(y,x,1)  % only consider pixels on foot
                pixcount=pixcount+1;
                yvalid(x,pixcount)=y;
                intensity(x,pixcount)=sum(pd(y,x,1:3));
                orgyidx(x,pixcount)=y;
            else
                firstpix(x)=firstpix(x)+1;
            end
        end
        if pixcount>npix_valid  % make sure that sufficient pixels are on foot in that vertical line
            intensity(x,1:pixcount)=smooth(intensity(x,1:pixcount));
            [dummy,idx]=min(intensity(x,1:pixcount),[],'omitnan');
            yidx(x)=idx+firstpix(x);
        else
            yidx(x)=NaN;
        end
    end
    %coeff=polyfit(xvalid(1:xcount),yidx(1:xcount),2);
    plot(1:nx,yidx,'+');
    %plot(polyval(coeff,xvalid(1:xcount)),'Color', [1 0 1], 'LineWidth',2);
    
    nfig=nfig+1;
    figure(nfig);
    hold on
    surf(1:maxwidx,1:nx,vert_intensity_fit);
%     for x=1:xcount
%         plot3(yidx(x),xvalid(x),vert_intensity_fit(xvalid(x),yidx(x)),'*');
%     end
    view(30,40);
end


%% plot histogram
if plot_histogram
    nfig=nfig+1;
    figure(nfig);
    hold on
    if plot_red_hist plot(clrdist(:,1),'r'); end;
    if plot_green_hist plot(clrdist(:,2),'g'); end;
    if plot_blue_hist plot(clrdist(:,3),'b'); end;
    title('Histogram of Pixel-Colours on Foot');
end
%% parameters for pressure detection algortihm

close_to_min=0.7;       % range ]0,1], how much of the interval green_max to green_min is used for first estimate
confidence_level=0.95;  % confidence level for classification, based on posterior_estimate of classify
reclassify=true;        % should raw classification be overruled

%% find two green peaks
if background 
    offset=1;
else
    offset=20;
end
median=find(intclrdist(:,2)>footpixcount/2,1,'first');
[maxn1 maxidx1]=max(clrdist(offset:median,2));  % starts from offset to suppress dark pixels from black background feet
max1idx=maxidx1+offset-1;
[maxn2 maxidx2]=max(clrdist(median:256,2));
maxidx2=maxidx2+median-1;
[minn minidx]=min(clrdist(maxidx1:maxidx2,2));
minidx=minidx+maxidx1-1;

line([maxidx1 maxidx1],[0 maxn1],'Color', [0.5 0.5 0.5]);
line([maxidx2 maxidx2],[0 maxn2],'Color', [0.5 0.5 0.5]);
line([minidx minidx],[0 minn],'Color', [0.5 0.5 0.5]);

%% define range around peaks

range1=round(close_to_min*(minidx-maxidx1));
range2=round(close_to_min*(maxidx2-minidx));

line([maxidx1+range1 maxidx1+range1],[0 clrdist(maxidx1+range1,2)],'Color', [0.7 0.7 0.7], 'LineStyle','--');
line([maxidx2+range2 maxidx2+range2],[0 clrdist(maxidx2+range2,2)],'Color', [0.7 0.7 0.7], 'LineStyle','--');
line([maxidx1-range1 maxidx1-range1],[0 clrdist(maxidx1-range1,2)],'Color', [0.7 0.7 0.7], 'LineStyle','--');
line([maxidx2-range2 maxidx2-range2],[0 clrdist(maxidx2-range2,2)],'Color', [0.7 0.7 0.7], 'LineStyle','--');

%transfer limits to 0 to 1 range as required for arrays of type 'double'
lower=(maxidx1+range1)/256;
lowerleft=(maxidx1-range1)/256;
upper=(maxidx2-range2)/256;
upperright=(maxidx2+range2)/256;

%% first estimate of pressure areas

rawpressmask=pblack;

for x=1:nx
    for y=1:ny
        if footmask2(y,x,1)
            if pd(y,x,2)>lowerleft && pd(y,x,2)<lower
                rawpressmask(y,x,3)=1;
            elseif pd(y,x,2)>upper && pd(y,x,2)<upperright
                rawpressmask(y,x,1)=1;
            else
                rawpressmask(y,x,2)=1;
            end
        end
    end
end
if plot_rawpressure
    nfig=nfig+1;
    figure(nfig);
    image (rawpressmask);
    axis equal
    title ('Pressured Areas (Raw)');
end

%% classification
c=zeros(nx*ny,3);
t=c;
tgroup=zeros(nx*ny,1);
cx=zeros(nx*ny,1);
cy=cx;
ccount=0;
tcount=0;

for x=1:nx
    for y=1:ny
        if footmask(y,x,1)                      % if pixel is on foot
            if rawpressmask(y,x,2)              % if pixel classification is unclear
                ccount=ccount+1;                    
                c(ccount,1:3)=pd(y,x,1:3);      % add to pixels requiring classification
                cx(ccount)=x;                   % save x-position of pixel
                cy(ccount)=y;                   % save y-position of pixel
            elseif rawpressmask(y,x,3)|| rawpressmask(y,x,1)
                tcount=tcount+1;
                t(tcount,1:3)=pd(y,x,1:3);      % add to pixels used for training
                if rawpressmask(y,x,3)
                    tgroup(tcount)=0;           % no pressure
                else
                    tgroup(tcount)=1;           % pressure
                end
                if reclassify                   % allow correction of raw classification
                    ccount=ccount+1;
                    c(ccount,1:3)=pd(y,x,1:3);
                    cx(ccount)=x;                   
                    cy(ccount)=y;    
                end                
            end
        end
    end
end

 % perform classification
[newclass,err,POSTERIOR,logp,coeff]=classify(c(1:ccount,1:3),t(1:tcount,1:3),tgroup(1:tcount)', 'quadratic');

finepressmask=rawpressmask;
for n=1:ccount
    if newclass(n)==1 && abs(POSTERIOR(n,1)-POSTERIOR(n,2))>confidence_level% pixel is assumed to be under pressure after classification
        finepressmask(cy(n),cx(n),1:3)=[1 0 0];
    else 
        finepressmask(cy(n),cx(n),1:3)=[0 0 1];
    end
end

if plot_finepressure
    nfig=nfig+1;
    figure(nfig);
    image (finepressmask);
    axis equal
    title ('Pressured Areas (Fine)');
end

%% plot changes by classification
diffpressmask=finepressmask | rawpressmask;
if plot_diffpressmasks
    nfig=nfig+1;
    figure(nfig);
    image (diffpressmask);
    axis equal
    title ('Changes by classification (Fine)');
end


%% classification by neural network  ... work in progress

if net_classification
    target=zeros(tcount,2);
    % reorganise grouping variable to format needed for neural network tools
    for k=1:tcount
        if tgroup(k)==0
            target(k,1)=1;
        else
            target(k,2)=1;
        end
    end

    hiddenLayerSize = 20;
    net = patternnet(hiddenLayerSize);

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 90/100;
    net.divideParam.valRatio = 5/100;
    net.divideParam.testRatio = 5/100;

    % train network
    [net,tr] = train(net,t(1:tcount,:)',target(:,:)');

    % apply network to pixels that need to be classified
    newclass=net(c(1:ccount,:)');
    safetymargin=0.8;  % classes must differ at least by this margin

    netpressmask=rawpressmask;
    for n=1:ccount
        if newclass(1,n)>newclass(2,n)+safetymargin  % pixel is assumed have no pressure after classification
            netpressmask(cy(n),cx(n),1:3)=[0 0 1];
        else
            netpressmask(cy(n),cx(n),1:3)=[1 0 0];
        end
    end

    if plot_netpressure
        nfig=nfig+1;
        figure(nfig);
        image (netpressmask);
        axis equal
        title ('Pressured Areas (Neural Net)');
    end
 
    %% plot changes by net_classification
    diffnetmask=netpressmask | rawpressmask;
    if plot_netpressmasks
        nfig=nfig+1;
        figure(nfig);
        image (diffnetmask);
        axis equal
        title ('Changes by Classification (Net)');
    end
end