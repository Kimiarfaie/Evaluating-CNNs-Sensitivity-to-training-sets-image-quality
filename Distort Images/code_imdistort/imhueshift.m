function [b] = hue_shift(a, factor)
hsvIm = rgb2hsv(a);
hsvIm(:,:,1) = hsvIm(:,:,1) + factor;
hsvIm(:,:,1) = mod(hsvIm(:,:,1), 1);
b = hsv2rgb(hsvIm);
end
