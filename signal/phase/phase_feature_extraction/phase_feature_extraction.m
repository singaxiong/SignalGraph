function [IF_phase_unwrap, GDD_phase, MGD_phase]=phase_feature_extraction(sp_complex, sp_delay)

% This function is using for different kinds of phase feature extraction,
% include instantaneous frequency deviation (IF), group delay deviation
% (GDD), modified group delay (MGD), phase distortion (PD), phase
% distortion standard deviation (PDD) and relative phase shift (RPS).
%
% input: 
%     sp_complex:   [n*d] n is frame number (time), d is the feature dimension (frequency)
%                   STFT of target waveform x(n) which contain both magnitude and phase information
%     sp_delay:     [n*d] n is frame number, d is the feature dimension 
%                   STFT of modified target waveform n*x(n)
%
% output:
%                   [n*d] n is frame number (time), d is the feature dimension (frequency)
%     IF_phase:     instantaneous frequency deviation
%     GDD_phase:    group delay deviation
%     MGD_phase:    modified group delay
%     PD_phase:     phase distortion
%     PDD_phase:    phase distortion standard deviation
%     RPS_phase:    relative phase shift
%

is_plot = 1;

phase = angle(sp_complex);
phase_unwrap = unwrap(phase);


% [~, IF_phase] = gradient(phase); % deviation along time aixs
[~, IF_phase_unwrap] = gradient(phase_unwrap); % deviation along time aixs

GDD_phase = group_delay_feature(sp_complex); % deviation along frequency aixs

MGD_phase = modified_group_delay_raw(sp_complex, sp_delay);

if is_plot
    figure
    subplot(5,1,1)
    imagesc(phase');colormap jet
    title('original phase')
    
    subplot(5,1,2)
    imagesc(phase_unwrap');colormap jet
    title('original unwraped phase')

    subplot(5,1,3)
    imagesc(IF_phase_unwrap');colormap jet
    title('instantaneous frequency deviation')

    subplot(5,1,4)
    gdd = GDD_phase.^ 0.4;
    imagesc(real(gdd'));colormap jet
    title('group delay deviation')

    subplot(5,1,5)
    imagesc(MGD_phase');colormap jet
    title('modified group delay')
end


% RPS_phase = relative_phase_shift(phase);
% 
% PD_phase = phase_distortion(phase);

% PDD_phase = phase_distortion_std_deviation(phase)