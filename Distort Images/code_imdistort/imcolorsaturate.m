function [b] = imcolorsaturate(a,factor)

hsvIm = rgb2hsv(a);
hsvIm(:,:,2) = hsvIm(:,:,2) * factor;
hsvIm(:,:,2) = min(max(hsvIm(:,:,2) * factor, 0), 1);
b = hsv2rgb(hsvIm);

end