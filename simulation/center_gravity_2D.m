function [R,V] = center_gravity_2D(g_matrix)

    gx = 0;
    gy = 0;
    avg = 0;
    
    [M,N]=size(g_matrix);
    for x=1:M
        for y=1:N
            gx = gx + x*g_matrix(x,y);
            gy = gy + y*g_matrix(x,y);
            avg = avg + g_matrix(x,y);
        end
    end
    
    gx = gx/avg;
    gy = gy/avg;
    
    V = gx;
    R = gy;
    
end