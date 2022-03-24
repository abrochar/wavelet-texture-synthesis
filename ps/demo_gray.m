% synthesis gray-level textures
clear all 

path('./matlabPyrTools', path)

% name = 'turb_zoom';
% name = 'cerise512';
% name = 'cerise';
% name = 'gravel';

% name = 'bricks';
% name = 'redpeppers';
% name = 'pebbles';
% name = 'wood';
% name = 'paisley3';

% name = 'woodso';
% name = 'feynmano';
% name = 'bulls';
% name = 'food0001co';
% name = 'food0008co';
% name = 'crowdo';
name = 'disco';

datname = name;   
load(sprintf('./data/%s.mat',datname))
im0 = double(img);

Nsc = 5;  % Number of scales
Nor = 8; % Number of orientations
Na = 9;  % Spatial neighborhood is Na x Na coefficients
	 % It must be an odd number!
    
Niter = 200;	% Number of iterations of synthesis loop
cmask = [1;1;1;1];
Nsx = size(im0,1); 	% Size of synthetic image is Nsy x Nsx
Nsy = size(im0,2);	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)
params = textureAnalysis(im0, Nsc, Nor, Na);

totalparams=length(params.pixelStats(:))+length(params.pixelLPStats(:))+...
    (Nsc+1)*(Na*Na+1)/2 + ... % length(params.autoCorrReal(:))+...
    Nsc*Nor*(Na*Na+1)/2 + ... % length(params.autoCorrMag(:))+...
    Nsc*Nor+2 + ... % length(params.magMeans(:))+...   
    Nsc*(Nor*(Nor-1))/2 + ... % length(params.cousinMagCorr(:))+...
    (Nsc-1)*Nor*Nor + ... % length(params.parentMagCorr(:))+...
    (Nsc-1)*Nor*(2*Nor) + (Nsc*Nor*Nor) + ... %     length(params.cousinRealCorr(:))+...length(params.parentRealCorr(:))+...
    length(params.varianceHPR);

fprintf('total number of params is %d\n',totalparams);

oname=sprintf('./results_gray/%s_ps_gray_J5L8Dn4.mat',name);
if 1
    res = textureSynthesis(params, [Nsy Nsx], Niter, cmask);
    im = res;
    save(oname,'im');
    name
    size(res)    
else
    fprintf('skip %s\n',oname)
end
