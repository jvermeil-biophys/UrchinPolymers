%% Fitting Jeffrey's Model to Single Bead Pulling Data Using fminsearch Optimization Method
%%
clearvars;
close all;
%clc;

%% Setting the Parameters 
directory = 'F:\good data\20240305_embPulling_1-2cell\20x_1umbeads_1fps_4_eggC_8_out';
name = '20x_1umbeads_1fps_4_8_pulling7-54_Tracks.xml';                             %name of trajectory file
output_name = 'output_p2';                                               %name of output file
output_folder = 'p2_new code1';                                          %name of output folder
cd(directory);	%Change current folder

initPullFrame = 6;                                                           %%% Previous frame to the magnet appear FROM FIJI MOVIE
finPullFrame = 20;                                                           %%% Final POSITION OF MAGNET FROM FIJI MOVIE
x_magnet = 548.5;                                                            %x magnet tip center in pixel                                                      
y_magnet = 844;                                                              %y magnet tip center in pixel

pixel_size = 0.451;                                                        %um/pix
time_stp = 1;                                                              %second
magnet_radius = 50;                                                        %um
bead_radius = 0.5;                                                         %um

v = @(x) 80.23*exp(-x/47.49)+1.03*exp(-x/22740.0);                         %magnet calibration function um/s
viscosity_glycerol = 0.0857;                                               %80% glycerol in Pa.s ! AT 21°C ! 

start1 = [2, 50, 900];                                                   %starting values of k gamma1 gamma2
start2 = [0.9 25];                                                       %starting values of a and tau


%% Loading and Selection of Trajectories
[tracks, md] = importTrackMateTracks([directory,'\',name]); % [tracks, metadata] = importTrackMateTracks(file)
[track_id, f1] = trackSelection(tracks); % Custom code ???
selected_track = tracks{track_id,1};  % {.,.} indexes a "cell array" in Matlab

X = selected_track(:,2);
Y = selected_track(:,3);
t = selected_track(:,1);

initPullTime=find(t==(initPullFrame-1));    %% THIS IS AN INDEX NOT THE VALUE OF THE CELL
finPullTime= find(t==(finPullFrame-1));     %% THIS IS AN INDEX NOT THE VALUE OF THE CELL


%% Preliminary Trajectory Check
if  selected_track(end,1)<finPullTime+3
    msg_box = msgbox({'Selected track starts after pulling!',...
        'Not enough points to fit relaxation curve!'}, 'Error', 'error');
    return
end

%% Rotating Selected Track and Visual Check

theta = atan2((Y(initPullTime)-Y(finPullTime)),(X(initPullTime)-X(finPullTime)));

rotation_mat = [cos(-theta) -sin(-theta);sin(-theta) cos(-theta)];

x_rot = X*rotation_mat(1,1) + Y*rotation_mat(1,2);
y_rot = X*rotation_mat(2,1) + Y*rotation_mat(2,2);

pull_index = (initPullTime:finPullTime);       


xp = x_rot(pull_index);
yp = y_rot(pull_index);

f2 = figure(2);
subplot(1,2,1)
plot(X,Y,'.-','linewidth',1);
title('original track'); axis equal;
set(gca, 'ydir', 'reverse');
subplot(1,2,2)
plot(x_rot,y_rot,'.-','linewidth',1);
title('rotated track'); axis equal; hold all;
p = plot(xp, yp,'k.-','linewidth',2);
legend(p, 'pulled segment');


%% Pulling Phase
d = [];
d(:,1) = (x_magnet-X(pull_index))*pixel_size;
d(:,2) = (y_magnet-Y(pull_index))*pixel_size;
dist = sqrt(d(:,1).^2+d(:,2).^2)-magnet_radius;                            %bead distance to magnet surface

pull_force = 6*pi*viscosity_glycerol*v(dist)*bead_radius;                  %magnetic pulling force in pN


x_shift = (-x_rot(initPullTime:end)+max(x_rot(initPullTime:end)))*pixel_size;   %flipped and shifted x coordinate

pull_length = (finPullTime) - (initPullTime) + 1;
dx.pulling = x_shift(pull_index-initPullTime+1);                  %%%%%% ADD +1 TO START AT THE FIRST PULLING POSITION
dx.pulling_n = dx.pulling./pull_force;                                     %scaled displacement 
tpulling = (t(pull_index)-t(initPullTime))'*time_stp;     
tpulling=tpulling';


% Fitting Jeffrey's model to the pulling phase
jeffery_model = @(k, gamma1, gamma2, x) (1-exp(-k*x/gamma1))/k + x/gamma2;

g1 = @(start1) norm(jeffery_model(start1(1), start1(2), start1(3), tpulling)-dx.pulling_n);

options = optimset('TolX', 1e-10, 'TolFun', 1e-10, 'MaxFunEvals',500);
fit1 = fminsearch(g1, start1, options);

k = fit1(1);
gamma1 = fit1(2);
gamma2 = fit1(3);

f3 = figure(3);
subplot(1, 2, 1);
plot(tpulling, dx.pulling_n, 's');
hold all; axis square;
time = linspace(0,tpulling(end),1000);
y1 = jeffery_model(k, gamma1, gamma2, time);
plot(time, y1, 'linewidth', 1.5,'color', 'r');
xlabel('t [s]');
ylabel('dx/f [\mum/pN]');

pulling.curves = [tpulling dx.pulling_n];
pulling.force = [tpulling dx.pulling pull_force];
%pulling.fitting = [k, gamma1, gamma2];
tempo1=gamma1/k; 
tempo2=gamma2/k;
pulling.fitting = [k, gamma1, gamma2, tempo1, tempo2];


%% Release Phase
% Fitting exponential model with an offset to the release curve
exp_fit = @(a, tau, x) (1-a)*exp(-x/tau)+a;

dx.release_n = x_shift((pull_length):end)/x_shift(pull_length);          %normalized release displacement

vx_release = diff(x_rot((finPullTime):end));        %% THIS IS NOT SPEED, IT IS DISPLACEMENT OF X_ROTATED IN THE RELEASING PHASE
vy_release = diff(y_rot((finPullTime):end));        %% THIS IS NOT SPEED, IT IS DISPLACEMENT OF Y_ROTATED IN THE RELEASING PHASE
alpha = atan2(vy_release,vx_release);                                      %angle between two succesive steps
%correlation = cos(diff(alpha));
%ind = find(correlation<0.1)+1;                                             %last relaxation step when particle moves backward

%dx.release_n = dx.release_n(1:ind);
%t.release = (0:ind-1)'*time_stp;

Finalindex=length(t);
Release_index = ((finPullTime):Finalindex);     %% THIS IS THE LENGTH OF THE RELEASING PHASE AND IS BASED ON FRAME SHAPE (not time, ie, starts in 1)
trelease = (t(Release_index)-t(finPullTime))'*time_stp;   %% RELEASING TIMES
trelease=trelease';

g2 = @(start2) norm(exp_fit(start2(1), start2(2), trelease)-dx.release_n);

options = optimset('TolX', 1e-7, 'TolFun', 1e-7, 'MaxFunEvals',1000);
fit2 = fminsearch(g2, start2, options);

a = fit2(1);
tau = fit2(2);

subplot(1, 2, 2);
plot(trelease, dx.release_n, 's');
hold all; axis square;
time = linspace(0,trelease(end),1000);
y2 = exp_fit(a, tau, time);
plot(time, y2, 'linewidth', 1.5,'color', 'r');
ylim([0 1.5]);
xlabel('t [s]');
ylabel('Normalized displacement');

release.curves = [trelease dx.release_n];
release.fitting = [a, tau];


%% Saving Results in Excel
mkdir(output_folder);
cd([directory, '/', output_folder]);

%first sheet for the puling phase
headers.pulling1 = {'t[s]', 'dx.pulling', 'Pulling Force [pN]'};
xlswrite(output_name, headers.pulling1, 1, 'B2');
xlswrite(output_name, pulling.force, 1, 'B3');

%second sheet for the pulling phase
headers.pulling = {'t[s]', 'dx/f[um/pN]', 'k[pN/um]','gamma1[pN.s/um]', 'gamma2[pN.s/um]', 'tempo1[s]', 'tempo2[s]'};
xlswrite(output_name, headers.pulling, 2, 'B2');
xlswrite(output_name, pulling.curves, 2, 'B3');
xlswrite(output_name, pulling.fitting, 2, 'D3');

%third sheet for the release phase
headers.release = {'t[s]', 'dx/dx(0)', 'a','tau[s]'};
xlswrite(output_name, headers.release, 3, 'B2');
xlswrite(output_name, release.curves, 3, 'B3');
xlswrite(output_name, release.fitting, 3, 'D3');


%% Saving Workplace and Figures
save([output_name, '.mat']);
saveas(f1,'selected_track.jpg');
saveas(f2,'trajectories.jpg');
saveas(f3,'fits.jpg');
%% Print in screen to copy
disp([num2str(k), ' ', num2str(gamma1), ' ', num2str(gamma2), ' ', num2str(a), ' ', num2str(tau)]);