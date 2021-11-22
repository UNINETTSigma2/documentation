//0 OMP parallel for
#pragma omp parallel for
{
    for (int y = 0; y < height; y++)
    {
#pragma omp parallel for
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}


//1 Target
#pragma omp target
{
    for (int y = 0; y < height; y++)
    {                                                                                                              
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}
//2 Teams
#pragma omp target teams
{
    for (int y = 0; y < height; y++) 
    {                                                                                                              
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}
//3 Target teams parallel for 
#pragma omp target teams
#pragma omp parallel for
{
    for (int y = 0; y < height; y++) 
    {                                                                                                              
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}
//4 Target teams parallel for collapse
#pragma omp target teams
#pragma omp parallel for collapse(2)
{
    for (int y = 0; y < height; y++) 
    {                                                                                                              
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}
//5 Target teams parallel for collapse distribute
#pragma omp target teams distribute parallel for collapse(2) schedule(static, 1)
{
    for (int y = 0; y < height; y++) 
    {                                                                                                              
        for (int x = 0; x < width; x++) 
        {
            const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
            image[y * width + x] = palette[iters % palette_size];
        }
    }
}
// 6 mapping
#pragma omp target data map(to:palette[0:palette_size]), map(from:image[0:width*height])  
{	
#pragma omp target teams distribute parallel for collapse(2) schedule(static, 1)
    {
        for (int y = 0; y < height; y++) 
        {                                                                                                              
            for (int x = 0; x < width; x++) 
            {
                const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
                image[y * width + x] = palette[iters % palette_size];
            }
        }
    }
}
