function lp = lp_filter(c)

    len = max(size(c));
    win = 3;
    
    f=fft(c);
    f(1) = 0;
    f(2+win:len-win) = 0;
    lp = ifft(f);

end