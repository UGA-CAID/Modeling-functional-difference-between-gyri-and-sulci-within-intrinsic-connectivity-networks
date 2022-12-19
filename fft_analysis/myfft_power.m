function  [Freq, Amp, Y] = myfft(ts, T, ii)
% Usage: [Freq, Amplitude, Y] = myfft(ts, T)
% input
% ts: one vector of time series 
% T: Sampling period
% output
% f: frequency

if size(ts, 1) == 1
    ts = ts';
end
if size(ts, 2) ~= 1
    error('ts should be a vector');
end
% T = 0.72;             % Sampling period
Fs = 1/T;            % Sampling frequency
L = size(ts, 1);             % Length of signal
N = 2^nextpow2(L);
%% Fourier analysis
Y = fft(ts, N); %Compute the Fourier transform of the signal.

Freq = Fs*(0:(N/2))/N; % frequency
Freq = Freq';
%% Amplitude
P2 = abs(Y/N).^2; % two-sided spectrum P2
Amp = P2(1:N/2+1);  % single-sided spectrum of each signal.
Amp(2:end-1) = 2*Amp(2:end-1); % single-sided spectrum of each signal.

Angle = angle(Y); %Angle
t = (0:L-1)*T;        % Time vector
% plot(t,ts)
% title('Signal')
% xlabel('t (seconds)')
% ylabel('Amplitude')
% figname = ['layer0_timeseries.png'];
% saveas(gcf,figname);
% 
% plot(Freq,Amp)
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% cd('fft')
% figname = ['layer0_fft_',num2str(ii),'.png'];
% saveas(gcf,figname);
% cd ..