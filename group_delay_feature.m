
function [grp_phase1] = group_delay_feature(file_name, num_coeff)


%addpath(genpath('bosaris_toolkit'));

[speech,fs]  = audioread(file_name);

frame_length = 50; %msec

frame_shift  = 25; %msec
NFFT         = 2048;
pre_emph     = true;

%%% Pre-emphasis + framing 
if (pre_emph)
  
    speech = filter([1 -0.97], 1, speech);
end
frame_length = round((frame_length/1000)*fs);
frame_shift = round((frame_shift/1000)*fs);
frames = enframe(speech, hamming(frame_length), frame_shift);

frame_num    = size(frames, 1);
frame_length = size(frames, 2);
delay_vector = [1:1:frame_length];
delay_matrix = repmat(delay_vector, frame_num, 1);

delay_frames = frames .* delay_matrix;

x_spec = fft(frames', NFFT);
y_spec = fft(delay_frames', NFFT);
x_spec = x_spec(1:NFFT/2+1, :);
y_spec = y_spec(1:NFFT/2+1, :);

temp_x_spec = abs(x_spec);

dct_spec = dct(medfilt1(log(temp_x_spec), 5));

grp_phase1 = (real(x_spec).*real(y_spec) + imag(y_spec) .* imag(x_spec)) ./(exp(abs(x_spec)).^ (2));
grp_phase1(isnan(grp_phase1)) = 0.0;


