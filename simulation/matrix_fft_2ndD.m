function [ Result_Range_FFT  ] = matrix_fft_2ndD(data_matrix) 


    [N,L] = size(data_matrix) ;
    Result_Range_FFT = zeros(N,L);

    for Chirp_index = 1:1:N
        RawData = data_matrix(Chirp_index,:);% - mean(data_matrix(Chirp_index,:)) ;
        Result_Range_FFT(Chirp_index,:) = fft(RawData);%.*hamming(L)');
    end

end
