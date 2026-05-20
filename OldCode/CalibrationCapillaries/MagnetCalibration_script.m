%% Description
% This code performs the fit of calibration functions (distance-to-velocity & distance-to-force) for a magnet.
% It requires .xml files containing the results of the tracking with TrackMate. 
% It can work with one or several files at once.
% The outputs are the plots (.png) 
% and .json files containing the analysed data and the parameters of fitted functions.

% Below are an empty template to run your own calibration
% and an example calibration, already filled to run on the example data.


%% Empty Template

clear all;

% 1. Main directories
mainDir = "";                           % Where the track.xml files are found
saveDir = "";                           % Where the data and plots will be saved

% 2. Global Data
expLabel = '';                          % The label for this condition - used as a prefix for saved data and plots
saveResults = true;                     % If you want to export results as a .json file
savePlots = true;                       % If you want to save the plots as a .png file
SCALE = 0.451;                          % Image scale - µm/pix
FPS = 5;                                % Frame per second - 1/s
Rb = 1*0.5;                             % Bead radius - µm
visco = 53.3;                           % Dynamic Viscosity of the fluid - mPa.s
                                        % NB: a precise viscosity is important! This implies a precise measure of
filesInfo = [];                         % -> Initialized as an empty list, do not modify

% 3. fileInfos
fI.fileName = '';
fI.MagX = 0; fI.MagY = 0; fI.MagR = 0; 
fI.CropX = 0; fI.CropY = 0;
filesInfo = [filesInfo, fI];

fI.fileName = '';
fI.MagX = 0; fI.MagY = 0; fI.MagR = 0; 
fI.CropX = 0; fI.CropY = 0;
filesInfo = [filesInfo, fI];

% 4. Run the main function
MagnetCalibration_main(mainDir, SCALE, FPS, Rb, visco, filesInfo, ...
                       saveDir, expLabel, saveResults, savePlots)

%% Example Calibration - Do not modify

clear all;

% 1. Main directories
mainDir = ".\ExampleData\Tracks";       % Where the track.xml files are found
saveDir = ".\ExampleData\Results_matlab";      % Where the data and plots will be saved

% 2. Data relative to one calibration condition 
% These should be the same for all movies done for a given condition.
expLabel = 'MyOne_Glycerol75%';         % The label for this condition - used as a prefix for saved data and plots
saveResults = true;                     % If you want to export results as a .json file
savePlots = true;                       % If you want to save the plots as a .png file
SCALE = 0.451;                          % Image scale - µm/pix
FPS = 5;                                % Frame per second - 1/s
Rb = 1*0.5;                             % Bead radius - µm
visco = 53.3;                           % Dynamic Viscosity of the fluid - mPa.s
                                        % NB: a precise viscosity is important! This implies a precise measure of
filesInfo = [];                         % -> Initialized as an empty list, do not modify

% 3. For each movie, tracking has been done with TrackMate 
% and the results has been exported in a .xml file.
% Below, you have to fill the infos relative to each file.
% i.   fI.fileName - File Name
% ii.  fI.MagX, fI.MagY, fI.MagR - Position of the magnet center & radius of the magnet tip, in pixels
% iii. fI.CropX, fI.CropY - If the film where the tracking was done has been croped 
%                           from the first image, put the cropping coordinates below 
%                           (top left corner of the croping rectangle).
% iv.  filesInfo = [filesInfo, fI] - Finally the array 'filesInfo is appended.

fI.fileName = '25-11-19_Capi04_FilmBF_5fps_1_CropInv_Tracks.xml';
fI.MagX = 154; fI.MagY = 497; fI.MagR = 234*0.5; 
fI.CropX = 790; fI.CropY = 0;
filesInfo = [filesInfo, fI];

fI.fileName = '25-11-19_Capi04_FilmBF_5fps_2_CropInv_Tracks.xml';
fI.MagX = 140; fI.MagY = 551; fI.MagR = 232 * 0.5;
fI.CropX = 715; fI.CropY = 1;
filesInfo = [filesInfo, fI];

fI.fileName = '25-11-19_Capi04_FilmBF_5fps_4_CropInv_Tracks.xml';
fI.MagX = 149; fI.MagY = 610; fI.MagR = 238 * 0.5;
fI.CropX = 723; fI.CropY = 0;
filesInfo = [filesInfo, fI];


% 4. Run the main function
MagnetCalibration_main(mainDir, SCALE, FPS, Rb, visco, filesInfo, ...
                       saveDir, expLabel, saveResults, savePlots)
