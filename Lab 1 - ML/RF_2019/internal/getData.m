function [ data_train, data_query, img_tr, img_te ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name}; % 10 classes
        img_tr = [];
        img_te = [];
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            img_tr{c} = imgIdx_tr;
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            img_te{c} = imgIdx_te;
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [frame{c,i}, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method
        id2id = [];
        for i = 1:15
            for c = 1:10
                [~,n] = size(desc_tr{c,i});
                for p = 1:n
                    id2id = [id2id [c; i; p]];
                end
            end
        end
        [desc_sel, sel] = vl_colsubset(cat(2,desc_tr{:}), 10e4); % Randomly select 100k SIFT descriptors for clustering
        desc_sel = single(desc_sel);
        % K-means clustering        
        % write your own codes here
        % ...
        k = 500;
        ds = transpose(desc_sel);
        [~, cen] = kmeans(ds, k, "Replicates",3);
        nn = dsearchn(ds, cen);
        figure(); 
        for n = 1:50
            nid = nn(n);
            d = transpose(ds(nid,:));
            cip = transpose(id2id(:, sel(nid)));
            c = cip(1);
            i = cip(2);
            p = cip(3);
            fr = vl_frame2oell(frame{c, i}(:,p));
            M = reshape(fr(3:6),2,2);
            point = [fr(1) fr(2)];
            delta = [norm(M(1,:)) norm(M(2,:))];
            min = point - delta;
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));                
            I = imread(fullfile(subFolderName,imgList(img_tr{c}(i)).name));
            subaxis(5,10,n,'SpacingVert',0,'MR',0);
            icr = imcrop(I, [min(1), min(2), delta(1)*2, delta(2)*2]);
            imshow(icr)
            drawnow;
        end
        disp('Encoding Images...')
        % Vector Quantisation
        % write your own codes here
        % ...
        % data_train = dsearchn(cen, ds); 
        figure();
        cnt = 1;
        for c = 1:length(classList)
            for i = 1:imgSel(1)
                ds = transpose(single(vl_colsubset(cat(2,desc_tr{c,i}), 10e4)));
                enc{c,i} = dsearchn(cen, ds);
                for j = 1:k
                    data_train(imgSel(1)*(c-1) + i, j) = sum(enc{c,i}==j);
                end
                data_train(imgSel(1)*(c-1) + i, k+1) = c;
            end
            for i = 1:2
                subFolderName = fullfile(folderName,classList{c});
                imgList = dir(fullfile(subFolderName,'*.jpg'));                
                I = imread(fullfile(subFolderName,imgList(img_tr{c}(i)).name));
                subaxis(length(classList),4,cnt,'SpacingVert',0,'MR',0);
                imshow(I);
                cnt = cnt+1;
                perm = randperm(size(frame{c, i},2));
                sel = perm(1:50) ;
                h1 = vl_plotframe(frame{c, i}(:, sel)) ;
                h2 = vl_plotframe(frame{c, i}(:, sel)) ;
                set(h1,'color','k','linewidth',3) ;
                set(h2,'color','y','linewidth',2) ; 
                subaxis(length(classList),4,cnt,'SpacingVert',0,'MR',0);
                histogram(enc{c,i}, 1:k);
                cnt = cnt+1;
                drawnow;
            end
        end
        figure();
        cnt = 1;
        for i = 1:2
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));                
            I = imread(fullfile(subFolderName,imgList(img_tr{c}(i)).name));
            subaxis(2,2,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            perm = randperm(size(frame{c, i},2));
            sel = perm(1:50) ;
            h1 = vl_plotframe(frame{c, i}(:, sel)) ;
            h2 = vl_plotframe(frame{c, i}(:, sel)) ;
            set(h1,'color','k','linewidth',3) ;
            set(h2,'color','y','linewidth',2) ; 
            subaxis(2,2,cnt,'SpacingVert',0,'MR',0);
            histogram(enc{c,i}, 1:k);
            cnt = cnt+1;
            drawnow;
        end        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [frame_te{c,i}, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation
        cnt = 1;
        figure();
        for c = 1:length(classList)
            for i = 1:imgSel(2)
                ds = transpose(single(vl_colsubset(cat(2,desc_te{c,i}), 10e4)));
                enc{c,i} = dsearchn(cen, ds);
                for j = 1:k
                    data_query(imgSel(2)*(c-1) + i, j) = sum(enc{c,i}==j);
                end
                data_query(imgSel(2)*(c-1) + i, k+1) = c;
            end
            for i = 1:2
                subFolderName = fullfile(folderName,classList{c});
                imgList = dir(fullfile(subFolderName,'*.jpg'));                
                I = imread(fullfile(subFolderName,imgList(img_te{c}(i)).name));
                subaxis(length(classList),4,cnt,'SpacingVert',0,'MR',0);
                imshow(I);
                cnt = cnt+1;
                perm = randperm(size(frame_te{c, i},2)) ;
                sel = perm(1:50) ;
                h1 = vl_plotframe(frame_te{c, i}(:, sel)) ;
                h2 = vl_plotframe(frame_te{c, i}(:, sel)) ;
                set(h1,'color','k','linewidth',3) ;
                set(h2,'color','y','linewidth',2) ; 
                subaxis(length(classList),4,cnt,'SpacingVert',0,'MR',0);
                histogram(enc{c,i}, 1:k);
                cnt = cnt+1;
                drawnow;
            end
        end
        figure();  
        cnt = 1;
        for i = 1:2
            subFolderName = fullfile(folderName,classList{c});
            I = imread(fullfile(subFolderName,imgList(img_te{c}(i)).name));
            subaxis(2,2,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            perm = randperm(size(frame_te{c, i},2)) ;
            sel = perm(1:50) ;
            h1 = vl_plotframe(frame_te{c, i}(:, sel)) ;
            h2 = vl_plotframe(frame_te{c, i}(:, sel)) ;
            set(h1,'color','k','linewidth',3) ;
            set(h2,'color','y','linewidth',2) ; 
            subaxis(2,2,cnt,'SpacingVert',0,'MR',0);
            histogram(enc{c,i}, 1:k);
            cnt = cnt+1;
            drawnow;
        end        
        % write your own codes here
        % ...
        
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

