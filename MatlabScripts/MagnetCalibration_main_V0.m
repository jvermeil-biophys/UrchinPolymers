% MagnetCalibration.m
% Translated from MagnetCalibration.py (Python -> MATLAB)
% Usage: run MagnetCalibration_main at the MATLAB prompt (or open this file and run sections).
%
% Notes / dependencies:
% - Requires Optimization Toolbox for lsqcurvefit (used for curve_fit). If unavailable,
%   replace lsqcurvefit calls with nlinfit or fit.
% - Uses csaps / fnval / fnder from Curve Fitting Toolbox for smoothing splines. If not
%   available, replace with spline + gradient or conv smoothing.
% - JSON read/write uses matlab's jsonencode/jsondecode.
% - XML parsing uses xmlread.
%
function MagnetCalibration_main(mainDir, SCALE, FPS, Rb, visco, filesInfo, ...
                                saveDir, expLabel, saveResults, savePlots)
    % Example main that mirrors the bottom of the original Python script.
    % setGraphicOptions('screen_big');

    % Example directories (user should change these paths)
    % mainDir = fullfile('C:','Users','Utilisateur','Desktop','AnalysisPulls', ...
    %     '25-11_DynabeadsInCapillaries_CalibrationsTests','Tracks');
    % saveDir = fullfile('C:','Users','josep','Desktop','Seafile','AnalysisPulls', ...
    %     '25-11_DynabeadsInCapillaries_CalibrationsTests');

    % Constants
    % Rb = 1 * 0.5; % bead radius
    % visco = 53.3; % mPa.s
    % SCALE = 0.451;
    % FPS = 5;

    tracks_data = {};
    Nfiles = length(filesInfo);
    for i=1:Nfiles
        fI = filesInfo(i);
        fileName = fI.fileName;
        filePath = fullfile(mainDir,fileName);
        MagX = fI.MagX; MagY = fI.MagY; MagR = fI.MagR;
        CropX = fI.CropX; CropY = fI.CropY;
        all_tracks = importTrackMateTracks(filePath);
        tracks_data = [tracks_data, tracks_pretreatment(all_tracks, SCALE, FPS, MagX, MagY, MagR, Rb, visco, CropX, CropY)];
    end

    % Run analysis
    tracks_analysis(tracks_data, expLabel, saveResults, savePlots, saveDir);

end

%% -------------------- Utility functions --------------------
function setGraphicOptions(mode)
    switch mode
        case 'screen'
            SMALLER_SIZE = 11; SMALL_SIZE = 13; MEDIUM_SIZE = 16; BIGGER_SIZE = 20;
        case 'screen_big'
            SMALLER_SIZE = 12; SMALL_SIZE = 14; MEDIUM_SIZE = 18; BIGGER_SIZE = 22;
        case 'print'
            SMALLER_SIZE = 8; SMALL_SIZE = 10; MEDIUM_SIZE = 11; BIGGER_SIZE = 12;
        otherwise
            SMALLER_SIZE = 11; SMALL_SIZE = 13; MEDIUM_SIZE = 16; BIGGER_SIZE = 20;
    end
    set(groot,'defaultAxesFontSize',SMALL_SIZE);
    set(groot,'defaultTextFontSize',SMALL_SIZE);
    set(groot,'defaultLegendFontSize',SMALLER_SIZE);
    set(groot,'defaultFigureTitleFontSize',BIGGER_SIZE);
end

function c = lighten_color(hex_or_rgb, factor)
    if ischar(hex_or_rgb)
        c = reshape(sscanf(hex_or_rgb(2:end),'%2x%2x%2x'),3,[])'/255;
    else
        c = hex_or_rgb(:)';
    end
    % convert rgb to hls-like via rgb2hsv approximation (MATLAB has rgb2hsv)
    hsv = rgb2hsv(c);
    % adjust value (v) as a proxy for luminosity
    hsv(3) = max(0,min(1,factor*hsv(3)));
    c = hsv2rgb(hsv);
end

%% -------------------- Fitting routines --------------------


function [params, stats] = fitLine(X, Y)
    % Linear OLS fit: Y = a*X + b. returns params = [b; a]
    Xmat = [ones(length(X),1), X(:)];
    b = Xmat\Y(:);
    params = b;
    % Basic stats struct
    residuals = Y(:) - Xmat*b;
    SSres = sum(residuals.^2);
    SStot = sum((Y(:)-mean(Y)).^2);
    R2 = 1 - SSres/SStot;
    stats.R2 = R2; stats.residuals = residuals;
end

function [params, stats] = fitLineHuber(X, Y)
    % Robust linear fit using Huber weighting. Returns params [b; a].
    Xmat = [ones(length(X),1), X(:)];
    [b,stats] = robustfit(X, Y, 'huber'); % robustfit returns [intercept; slope]
    params = b;
    % compute weights and run weighted least squares (WLS)
    w = stats.w;
    W = diag(w);
    wlm_params = (Xmat' * W * Xmat) \ (Xmat' * W * Y(:));
    residuals = Y(:) - Xmat*wlm_params;
    SSres = sum(residuals.^2);
    SStot = sum((Y(:)-mean(Y)).^2);
    R2 = 1 - SSres/SStot;
    stats.R2 = R2; stats.residuals = residuals;
end

function y = doubleExpo(x, A, k1, B, k2)
    y = A.*exp(-x./k1) + B.*exp(-x./k2);
end

function y = powerLaw(x, A, k)
    y = A .* (x.^k);
end

function popt = fitPowerLaw(X, Y, p0, lb, ub)
    ft = fittype('A*x^k','independent','x','coefficients',{'A','k'});
    opts = fitoptions(ft);
    opts.StartPoint = p0;
    opts.Lower = lb;
    opts.Upper = ub;
    mdl = fit(X, Y, ft, opts);
    popt = [mdl.A, mdl.k];
end

function popt = fitDoubleExpo(X, Y, p0, lb, ub)
    ft = fittype('A*exp(-x/k1) + B*exp(-x/k2)', ...
             'independent','x', ...
             'coefficients',{'A','k1','B','k2'});
    opts = fitoptions(ft);
    opts.StartPoint = p0;
    opts.Lower = lb;
    opts.Upper = ub;
    mdl = fit(X, Y, ft, opts);
    popt = [mdl.A, mdl.k1, mdl.B, mdl.k2];
end

%% -------------------- JSON helpers --------------------
function dict2json(s, dirPath, fileName)
    if ~exist(dirPath,'dir'), mkdir(dirPath); end
    jsonText = jsonencode(s,PrettyPrint=true);
    fid = fopen(fullfile(dirPath,[fileName '.json']),'w');
    fwrite(fid,jsonText,'char'); fclose(fid);
end

function s = json2dict(dirPath, fileName)
    txt = fileread(fullfile(dirPath,[fileName '.json']));
    s = jsondecode(txt);
end

function listOfdict2json(L, dirPath, fileName)
    if ~exist(dirPath,'dir'), mkdir(dirPath); end
    jsonText = jsonencode(L,PrettyPrint=true);
    fid = fopen(fullfile(dirPath,[fileName '.json']),'w');
    fwrite(fid,jsonText,'char'); fclose(fid);
end

function L = json2listOfdict(dirPath, fileName)
    txt = fileread(fullfile(dirPath,[fileName '.json']));
    L = jsondecode(txt);
end

%% -------------------- XML import --------------------
function tracks = importTrackMateTracks(filepath)
    % Parse TrackMate XML file and return cell array of tracks (Nx3 [t x y])
    if ~exist(filepath,'file')
        error('File not found: %s', filepath);
    end
    xDoc = xmlread(filepath);
    particles = xDoc.getElementsByTagName('particle');
    tracks = {};
    for i=0:particles.getLength-1
        particle = particles.item(i);
        detections = particle.getElementsByTagName('detection');
        L = [];
        for j=0:detections.getLength-1
            det = detections.item(j);
            t = str2double(char(det.getAttribute('t')));
            x = str2double(char(det.getAttribute('x')));
            y = str2double(char(det.getAttribute('y')));
            L = [L; t, x, y];
        end
        tracks{end+1} = L; %#ok<AGROW>
    end
end

%% -------------------- Track cleaning & pretreatment --------------------
function [cleaned_track, track_valid] = cleanRawTrack(track)
    track_valid = true;
    cleaned_track = track;
    N = size(track,1);
    X = track(:,2); Y = track(:,3);
    mX = X==min(X); MX = X==max(X);
    mY = Y==min(Y); MY = Y==max(Y);
    NmX = sum(mX); NMX = sum(MX); NmY = sum(mY); NMY = sum(MY);
    max_saturation = max([NmX, NMX, NmY, NMY]);
    if max_saturation >= 0.8*N || (N - max_saturation) < 20
        track_valid = false;
    elseif max_saturation > 2
        [~,i] = max([NmX, NMX, NmY, NMY]);
        filters = {~mX, ~MX, ~mY, ~MY};
        filter_array = filters{i};
        cleaned_track = cleaned_track(filter_array,:);
    end
end

function clean_tracks = cleanAllRawTracks(all_tracks)
    clean_tracks = {};
    for i=1:length(all_tracks)
        track = all_tracks{i};
        [cleaned_track, track_valid] = cleanRawTrack(track);
        if track_valid
            clean_tracks{end+1} = cleaned_track; %#ok<AGROW>
        end
    end
end

function tracks_data = tracks_pretreatment(all_tracks, SCALE, FPS, MagX, MagY, MagR, Rb, visco, CropX, CropY)
    MagX = MagX * SCALE; MagY = MagY * SCALE; MagR = MagR * SCALE;
    CropX = CropX * SCALE; CropY = CropY * SCALE;
    all_tracks = cleanAllRawTracks(all_tracks);
    tracks_data = {};
    for i=1:length(all_tracks)
        track = all_tracks{i};
        T = track(:,1) * (1/FPS);
        X = track(:,2) * SCALE;
        Y = track(:,3) * SCALE;
        tdat.T = T; tdat.Xr = X; tdat.Yr = Y;

        % Coordinate relative to magnet
        X2 = (X + CropX) - MagX;
        Y2 = MagY - (Y + CropY);
        medX2 = median(X2); medY2 = median(Y2);
        tdat.X = X2; tdat.Y = Y2; tdat.medX = medX2; tdat.medY = medY2;

        % Fit of the trajectory to assess angle & smoothness
        %[parms, stats] = fitLineHuber(X2, Y2);
        [parms, stats] = fitLine(X2, Y2);
        b_fit = parms(1); a_fit = parms(2);
        R2 = stats.R2;
        theta = atan(a_fit);
        tdat.a_fit = a_fit; tdat.b_fit = b_fit; 
        tdat.r2_fit = R2; tdat.theta = theta;

        % Compute the angle of the median position with respect to the
        % magnet center
        phi = atan(medY2/medX2);
        delta = theta - phi;
        tdat.phi = phi; tdat.delta = delta;

        % Define the distance, smooth and derive the velocity
        D = sqrt(X2.^2 + Y2.^2) - MagR;
        pp = csaps(T, D, 1e-3);
        dpp = fnder(pp,1);
        V_spline = abs(fnval(dpp, T));
        V = V_spline;

        % Compute the force
        F = 6*pi*visco*1e-3 * Rb*1e-6 .* V*1e-6 * 1e12; % pN
        medD = median(D); medV = median(V); medF = median(F);
        tdat.D = D; tdat.V = V; tdat.F = F; 
        tdat.medD = medD; tdat.medV = medV; tdat.medF = medF;

        % Append the result cell array
        tracks_data{i} = tdat;
    end
end


%% -------------------- Analysis functions --------------------
function tracks_analysis(tracks_data, expLabel, saveResults, savePlots, saveDir)
    MagR = 60; % µm
    % Colors
    c_V = '#1f77b4'; % Strong blue
    c_F = '#2ca02c'; % Dark lime green
    c_excluded = '#ff7f0e'; % Vivid orange
    c_fit2exp = '#ff0000'; % Red
    c_fitpL = '#ff8c00'; % Dark orange

    % First filter
    tracks_data_f1 = {};
    for i=1:length(tracks_data)
        track = tracks_data{i};
        crit1 = abs(track.delta*180/pi) < 25;
        crit2 = (track.r2_fit > 0.80);
        bypass1 = min(track.X) < 300;
        if (crit1 && crit2) || bypass1
            tracks_data_f1{end+1} = track; %#ok<AGROW>
        end
    end

    fig1 = figure('Name','Trajectories');
    fig1.Position = [0, 100, 1800, 500];
    ax1 = subplot(1,3,1);
    title(ax1, sprintf('All tracks, N = %d', length(tracks_data)));
    hold(ax1,'on');
    for i=1:length(tracks_data)
        plot(ax1, tracks_data{i}.X, tracks_data{i}.Y);
    end
    viscircles([0,0], MagR, 'Color',[0.5 0.5 0.5]);
    xlim(ax1,[0,800]); ylim(ax1,[-400,400]); grid(ax1,'on'); axis(ax1,'equal');
    xlabel(ax1,'X [µm]'); ylabel(ax1,'Y [µm]');
    hold(ax1,'off');

    ax2 = subplot(1,3,2);
    title(ax2, sprintf('First filter, N = %d', length(tracks_data_f1)));
    hold(ax2,'on');
    for i=1:length(tracks_data_f1)
        plot(ax2, tracks_data_f1{i}.X, tracks_data_f1{i}.Y);
    end
    viscircles([0,0], MagR, 'Color',[0.5 0.5 0.5]);
    xlim(ax2,[0,800]); ylim(ax2,[-400,400]); grid(ax2,'on'); axis(ax2,'equal');
    xlabel(ax2,'X [µm]'); ylabel(ax2,'Y [µm]');
    hold(ax2,'off');
    % for ax = [ax1, ax2]
    %     hold(ax,'on');
    %     % Do sth
    %     hold(ax,'off');
    % end

    % Concatenate D and V
    all_medD = cellfun(@(t) t.medD, tracks_data_f1);
    all_medV = cellfun(@(t) t.medV, tracks_data_f1);
    % all_D = cell2mat(cellfun(@(t) t.D, tracks_data_f1,'UniformOutput',false));
    % all_V = cell2mat(cellfun(@(t) t.V, tracks_data_f1,'UniformOutput',false));
    Dcells = cellfun(@(t) t.D(:), tracks_data_f1, 'UniformOutput', false);
    all_D = vertcat(Dcells{:});
    Vcells = cellfun(@(t) t.V(:), tracks_data_f1, 'UniformOutput', false);
    all_V = vertcat(Vcells{:});

    D_plot = linspace(1,5000,500);

    % Fit power law to obtain naive fit
    % Use lsqcurvefit for power law
    % try
    % opts = optimoptions('lsqcurvefit','Display','off');
    p0 = [1000, -2]; lb = [0, -10]; ub = [Inf, 0];
    % fun = @(p,x) powerLaw(x,p(1),p(2));
    % V_popt_pL = lsqcurvefit(fun, p0, all_D, all_V, lb, ub); % , opts
    V_popt_pL = fitPowerLaw(all_D, all_V, p0, lb, ub);
    % catch
    %     % fallback to simple polyfit on log-log
    %     idx = all_D>0 & all_V>0;
    %     pp = polyfit(log(all_D(idx)), log(all_V(idx)), 1);
    %     V_popt_pL = [exp(pp(2)), pp(1)]; % approximate
    % end
    V_fit_pL = powerLaw(D_plot, V_popt_pL(1), V_popt_pL(2));

    expected_medV = powerLaw(all_medD, V_popt_pL(1), V_popt_pL(2));
    ratio_fitV = all_medV ./ expected_medV;
    high_cut = 1.45; low_cut = 0.55;

    tracks_data_f2 = {}; removed_tracks = {};
    for i=1:length(tracks_data_f1)
        if ratio_fitV(i) > low_cut && ratio_fitV(i) < high_cut
            tracks_data_f2{end+1} = tracks_data_f1{i};
        else
            removed_tracks{end+1} = tracks_data_f1{i};
        end
    end

    % all_D = cell2mat(cellfun(@(t) t.D, tracks_data_f2,'UniformOutput',false));
    % all_V = cell2mat(cellfun(@(t) t.V, tracks_data_f2,'UniformOutput',false));
    % all_F = cell2mat(cellfun(@(t) t.F, tracks_data_f2,'UniformOutput',false));
    % all_removedD = cell2mat(cellfun(@(t) t.D, removed_tracks,'UniformOutput',false));
    % all_removedV = cell2mat(cellfun(@(t) t.V, removed_tracks,'UniformOutput',false));
    Dcells = cellfun(@(t) t.D(:), tracks_data_f2, 'UniformOutput', false);
    all_D = vertcat(Dcells{:});
    Vcells = cellfun(@(t) t.V(:), tracks_data_f2, 'UniformOutput', false);
    all_V = vertcat(Vcells{:});
    Fcells = cellfun(@(t) t.F(:), tracks_data_f2, 'UniformOutput', false);
    all_F = vertcat(Fcells{:});
    rDcells = cellfun(@(t) t.D(:), removed_tracks, 'UniformOutput', false);
    all_removedD = vertcat(rDcells{:});
    rVcells = cellfun(@(t) t.V(:), removed_tracks, 'UniformOutput', false);
    all_removedV = vertcat(rVcells{:});
    

    % Plot second filter
    ax3 = subplot(1,3,3);
    a = 0.05;
    hold(ax3,'on');
    scatter(ax3, all_D, all_V, 'filled', 'Marker','o','SizeData',10, 'Color',c_V, 'MarkerFaceAlpha',a);
    plot(ax3, D_plot, V_fit_pL, '-.', 'LineWidth',1.5);
    plot(ax3, D_plot, V_fit_pL*high_cut, '-.', 'LineWidth',1.25);
    plot(ax3, D_plot, V_fit_pL*low_cut, '-.', 'LineWidth',1.25);
    scatter(ax3, all_removedD, all_removedV, 'filled', 'Marker','o','SizeData',10, 'Color',c_excluded, 'MarkerFaceAlpha',a);
    MD = max(all_D); MV = max(all_V);
    xlim(ax3,[0,1.1*MD]); ylim(ax3,[0,1.2*MV]); grid(ax3,'on');
    legend(ax3,'Data','Naive fit','High cut','Low cut');
    xlabel(ax3,'D [µm]'); ylabel(ax3,'V [µm/s]');
    title(ax3, sprintf('Second Filter, N = %d', length(tracks_data_f2)));
    hold(ax3,'off');

    % Final fits: Velocity and Force
    D_plot = linspace(1,5000,500);
    % fun_2exp = @(p,x) doubleExpo(x,p(1),p(2),p(3),p(4));
    % fun_pL = @(p,x) powerLaw(x,p(1),p(2));
    % opts = optimoptions('lsqcurvefit','Display','off');

    % Fits double exp
    p0 = [1000, 50, 100, 1000]; lb = [0,0,0,0]; ub = [Inf,Inf,Inf,Inf];

    % V_popt_2exp = lsqcurvefit(fun_2exp, p0, all_D, all_V, lb, ub); % opts
    V_popt_2exp = fitDoubleExpo(all_D, all_V, p0, lb, ub);
    V_fit_2exp = doubleExpo(D_plot, V_popt_2exp(1), V_popt_2exp(2), V_popt_2exp(3), V_popt_2exp(4));

    % F_popt_2exp = lsqcurvefit(fun_2exp, p0, all_D, all_F, lb, ub); % opts
    F_popt_2exp = fitDoubleExpo(all_D, all_F, p0, lb, ub);
    F_fit_2exp = doubleExpo(D_plot, F_popt_2exp(1), F_popt_2exp(2), F_popt_2exp(3), F_popt_2exp(4));

    % Fits power law
    p0 = [1000, -2]; lb = [0,-10]; ub = [Inf,0];

    % V_popt_pL = lsqcurvefit(fun_pL, p0, all_D, all_V, lb, ub); % opts
    V_popt_pL = fitPowerLaw(all_D, all_V, p0, lb, ub);
    V_fit_pL = powerLaw(D_plot, V_popt_pL(1), V_popt_pL(2));

    % F_popt_pL = lsqcurvefit(fun_pL, p0, all_D, all_F, lb, ub); % opts
    F_popt_pL = fitPowerLaw(all_D, all_F, p0, lb, ub);
    F_fit_pL = powerLaw(D_plot, F_popt_pL(1), F_popt_pL(2));

    % Plot fits
    fig2 = figure('Name','Fits');
    fig2.Position = [0, 0, 1600, 1200];
    figTiles = tiledlayout(2, 5, "TileSpacing", "tight");
    a = 0.05;

    % Plot V
    ax = nexttile([1 2]);
    scatter(ax, all_D, all_V, 'filled', 'Marker','o','SizeData',10, 'Color',c_V,'MarkerFaceAlpha',a); hold(ax,'on');
    plot(ax, D_plot, V_fit_2exp, '-', 'Color',c_fit2exp, 'LineWidth',1.5);
    plot(ax, D_plot, V_fit_pL, '-', 'Color',c_fitpL, 'LineWidth',1.5);
    grid(ax,'on'); xlim(ax,[0, 1.1*max(all_D)]); ylim(ax,[0, 1.2*max(all_V)]); xlabel(ax,'d [µm]'); ylabel(ax,'v [µm/s]');
    hold(ax,'off');

    ax = nexttile([1 2]);
    scatter(ax, all_D, all_V, 'filled', 'Marker','o','SizeData',10, 'Color',c_V,'MarkerFaceAlpha',a); hold(ax,'on');
    plot(ax, D_plot, V_fit_2exp, '-', 'Color',c_fit2exp, 'LineWidth',1.5);
    plot(ax, D_plot, V_fit_pL, '-', 'Color',c_fitpL, 'LineWidth',1.5);
    grid(ax,'on'); xlim(ax,[50,5000]); ylim(ax,[0.01,100]); xlabel(ax,'d [µm]'); ylabel(ax,'v [µm/s]');
    set(ax,'XScale','log','YScale','log');

    % Legend V
    legendText = sprintf('$$\\bf{Velocity}$$');
    legendText_2exp = sprintf(['$$\\mathbf{A \\cdot e^{-x/k_1} + B \\cdot e^{-x/k_2}}$$\n' ...
                              '$$A = %.2e,\\; k_1 = %.2f$$\n' ...
                              '$$B = %.2e,\\; k_2 = %.2f$$'], ...
                              V_popt_2exp(1), V_popt_2exp(2), V_popt_2exp(3), V_popt_2exp(4));
    legendText_pL = sprintf(['$$\\mathbf{A \\cdot x^k}$$\n' ...
                            '$$A = %.2e,\\; k = %.2f$$'], ...
                            V_popt_pL(1), V_popt_pL(2));
    lgd = legend(legendText,legendText_2exp,legendText_pL,'Interpreter','latex','Location','west','FontSize',11);
    lgd.Layout.Tile = 5;
    
    % Plot F
    ax = nexttile([1 2]);
    scatter(ax, all_D, all_F, 'filled', 'Marker','o','SizeData',10, 'Color',c_F,'MarkerFaceAlpha',a); hold(ax,'on');
    plot(ax, D_plot, F_fit_2exp, '-', 'Color',c_fit2exp, 'LineWidth',1.5);
    plot(ax, D_plot, F_fit_pL, '-', 'Color',c_fitpL, 'LineWidth',1.5);
    grid(ax,'on'); xlim(ax,[0, 1.1*max(all_D)]); ylim(ax,[0, 1.2*max(all_F)]); xlabel(ax,'d [µm]'); ylabel(ax,'F [pN]');

    ax = nexttile([1 2]);
    scatter(ax, all_D, all_F, 'filled', 'Marker','o','SizeData',10, 'Color',c_F,'MarkerFaceAlpha',a); hold(ax,'on');
    plot(ax, D_plot, F_fit_2exp, '-', 'Color',c_fit2exp, 'LineWidth',1.5);
    plot(ax, D_plot, F_fit_pL, '-', 'Color',c_fitpL, 'LineWidth',1.5);
    grid(ax,'on'); xlim(ax,[50,5000]); ylim(ax,[0.01,100]); xlabel(ax,'d [µm]'); ylabel(ax,'F [pN]');
    set(ax,'XScale','log','YScale','log');

    % Legend F
    legendText = sprintf('$$\\bf{Force}$$');
    legendText_2exp = sprintf(['$$\\mathbf{A \\cdot e^{-x/k_1} + B \\cdot e^{-x/k_2}}$$\n' ...
                      '$$A = %.2e,\\; k_1 = %.2f$$\n' ...
                      '$$B = %.2e,\\; k_2 = %.2f$$'], ...
                      F_popt_2exp(1), F_popt_2exp(2), F_popt_2exp(3), F_popt_2exp(4));
    legendText_pL = sprintf(['$$\\mathbf{A \\cdot x^k}$$\n' ...
                        '$$A = %.2e,\\; k = %.2f$$'], ...
                        F_popt_pL(1), F_popt_pL(2));
    lgd = legend(legendText,legendText_2exp,legendText_pL,'Interpreter','latex','Location','west','FontSize',11);
    lgd.Layout.Tile = 10;
    
    % Global Title
    sgtitle(['Calibration Data - ' expLabel], 'Interpreter', 'none');
    

    if savePlots
        print(fig1, fullfile(saveDir, [expLabel '_Traj.png']), "-dpng", '-r400');
        print(fig2, fullfile(saveDir, [expLabel '_Fits.png']), "-dpng", '-r400');
    end

    if saveResults
        dictResults.V_popt_pL = V_popt_pL;
        dictResults.F_popt_pL = F_popt_pL;
        dictResults.all_D = all_D; dictResults.all_V = all_V; dictResults.all_F = all_F;
        listOfdict2json(tracks_data_f2, saveDir, [expLabel '_allTracksData']);
        dict2json(dictResults, saveDir, [expLabel '_fitData']);
    end

end

%% -------------------- Compare analysis --------------------
function compareCalibration(srcDir, labelList, savePlots, saveDir)
    dataList = {};
    supTitle = '';
    saveTitle = '';
    for i=1:length(labelList)
        fitData = json2dict(srcDir, [labelList{i} '_fitData']);
        dataList{end+1} = fitData; 
        supTitle = [supTitle labelList{i} ' vs. ']; 
        saveTitle = [saveTitle labelList{i} '-v-'];
    end
    supTitle = supTitle(1:end-5); 
    saveTitle = saveTitle(1:end-3);

    fig1 = figure('Name','Compare');
    ax = gobjects(2,2);
    ax(1,1) = subplot(2,2,1); 
    ax(1,2) = subplot(2,2,2); 
    ax(2,1) = subplot(2,2,3); 
    ax(2,2) = subplot(2,2,4);

    D_plot = linspace(1,5000,500);

    for i=1:length(dataList)
        data = dataList{i}; lab = labelList{i};
        all_D = data.all_D; all_V = data.all_V; all_F = data.all_F;
        V_fit_2exp = doubleExpo(D_plot, data.V_popt_2exp(1), data.V_popt_2exp(2), data.V_popt_2exp(3), data.V_popt_2exp(4));
        F_fit_2exp = doubleExpo(D_plot, data.F_popt_2exp(1), data.F_popt_2exp(2), data.F_popt_2exp(3), data.F_popt_2exp(4));
        V_fit_pL = powerLaw(D_plot, data.V_popt_pL(1), data.V_popt_pL(2));
        F_fit_pL = powerLaw(D_plot, data.F_popt_pL(1), data.F_popt_pL(2));
        a = 0.05;

        axes(ax(1,1)); hold on; 
        scatter(all_D, all_V, 'filled', 'Marker','o','SizeData',10, 'MarkerFaceAlpha',a);
        plot(D_plot, V_fit_2exp,'--'); 
        plot(D_plot, V_fit_pL, ':'); 
        xlabel('d [µm]'); ylabel('v [µm/s]');

        axes(ax(1,2)); set(gca,'XScale','log','YScale','log'); hold on; 
        scatter(all_D, all_V, 'filled', 'Marker','o','SizeData',10, 'MarkerFaceAlpha',a);
        plot(D_plot, V_fit_2exp,'--'); 
        plot(D_plot, V_fit_pL,':'); 
        xlabel('d [µm]'); ylabel('v [µm/s]');

        axes(ax(2,1)); hold on; 
        scatter(all_D, all_F, 'filled', 'Marker','o','SizeData',10, 'MarkerFaceAlpha',a); 
        plot(D_plot, F_fit_2exp,'--'); 
        plot(D_plot, F_fit_pL,':'); 
        xlabel('d [µm]'); ylabel('F [pN]');

        axes(ax(2,2)); set(gca,'XScale','log','YScale','log'); hold on; 
        scatter(all_D, all_F, 'filled', 'Marker','o','SizeData',10, 'MarkerFaceAlpha',a); 
        plot(D_plot, F_fit_2exp,'--'); 
        plot(D_plot, F_fit_pL,':'); 
        xlabel('d [µm]'); ylabel('F [pN]');
    end

    sgtitle(['Calibration Data - ' supTitle]);
    if savePlots
        saveas(fig1, fullfile(saveDir, ['Compare_' saveTitle '.png'])); 
    end
end
