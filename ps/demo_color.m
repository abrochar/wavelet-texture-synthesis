addpath ./textureSynth-master
path('./matlabPyrTools', path)

% name = 'splatter';
% name = 'bricks';
% name = 'flower_bed';
% name = 'pebbles';

% name = 'cerise';
% name = 'bubbly';
% name = 'flowers';
% name = 'cerise512';
% name = 'gravel';
% name = 'pebbles';
name = 'bricks';

lmat = load(sprintf('./data/%s.mat',name));

Nsx = 256;  % Synthetic image dimensions
Nsy = 256;

im0 = uint8(lmat.im*255);
im0 = permute(im0,[2,3,1]);

syn = 2;

if syn==1
  Nsc = 5; % Number of pyramid scales
  Nor = 4; % Number of orientations
  Na = 5; % Number of spatial neighbors considered for spatial correlations
  Niter = 100; % Number of iterations of the synthesis loop
  cmask = [1,1,1,1,0];
  [params] = textureColorAnalysis(im0, Nsc, Nor, Na);
  tic; im = textureColorSynthesis(params, [Nsy Nsx], Niter, cmask); toc;

else
  Nsc = 5; % Number of pyramid scales
  Nor = 8; % Number of orientations
  Na = 9; % Number of spatial neighbors considered for spatial correlations
  Niter = 200; % Number of iterations of the synthesis loop
  cmask = [1,1,1,1,0];
  [params] = textureColorAnalysis(im0, Nsc, Nor, Na);
  totalparams=length(params.pixelStats(:))+...
              length(params.pixelStatsPCA(:))+...
              length(params.pixelLPStats(:))+...
              (Nsc+2)*3*(Na*Na+1)/2+...
              Nsc*Nor*(Na*Na+1)/2*3+...
              (Nsc*Nor+2)*3+...
              (3*Nor)*(3*Nor-1)/2*Nsc+...
              (Nsc-1)*((3*Nor)^2)+...
              Nsc*((3*Nor)^2)+...
              (Nsc-1)*(3*Nor)*(2*3*Nor)+3+6;
  fprintf('total number of params is %d\n',totalparams);

  tic; im = textureColorSynthesis(params, [Nsy Nsx], Niter, cmask); toc;
  im = permute(im,[3,1,2]);
  im = double(im)/255;
%  size(im)
end

if syn==1
  figure(1)
  subplot(121)
  imagesc(im0)
  axis square
  subplot(122)
  imagesc(im)
  axis square
end

if syn==1
  save(sprintf('./results/%s_%d_%d%d%d.mat',name,Nsx,Nsc,Nor,Na),'im');
else
  save(sprintf('./results_color/%s_pscolor%d%d%d%d.mat',name,Nsx,Nsc,Nor,Na),'im');
end
