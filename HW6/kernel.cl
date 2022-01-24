__kernel void convolution(int filterWidth, __constant float* filter, int imageHeight, int imageWidth, __global float* inputImage, __global float* outputImage)
{
    int halffilterSize = filterWidth / 2;
    int sum = 0;
    int k, l;
    int i = get_global_id(0);
    int j = get_global_id(1);
    bool inside = true;

    if(i <= halffilterSize || j <= halffilterSize || i > imageHeight-halffilterSize || j > imageWidth-halffilterSize)
        inside = false;

    if(inside){
        for (k = -halffilterSize; k <= halffilterSize; k++)
        {
            for (l = -halffilterSize; l <= halffilterSize; l++)
            {
                sum += inputImage[(i + k) * imageWidth + j + l] *
                       filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
            }
        }
    }
    else{
        for (k = -halffilterSize; k <= halffilterSize; k++)
        {
            for (l = -halffilterSize; l <= halffilterSize; l++)
            {
                if (i + k >= 0 && i + k < imageHeight &&
                    j + l >= 0 && j + l < imageWidth)
                {
                    sum += inputImage[(i + k) * imageWidth + j + l] *
                           filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
                }
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}
// __kernel void convolution(int filterWidth, __constant float* filter, int imageHeight, int imageWidth, __global float* inputImage, __global float* outputImage)
// {
//     int index = get_global_id(0);
//     int row = index /imageWidth;
//     int col = index % imageWidth;
//     int halffilterSize = filterWidth / 2;
//     int k, l;
//     float sum = 0.0f;
//
//     for (k = -halffilterSize; k <= halffilterSize; k++)
//     {
//         for (l = -halffilterSize; l <= halffilterSize; l++)
//         {
//             if(filter[(k + halffilterSize) * filterWidth + l + halffilterSize] != 0)
//             {
//                 if (row + k >= 0 && row + k < imageHeight &&
//                     col + l >= 0 && col + l < imageWidth)
//                 {
//                     sum += inputImage[(row + k) * imageWidth + col + l] *
//                             filter[(k + halffilterSize) * filterWidth +
//                                     l + halffilterSize];
//                 }
//             }
//         }
//     }
//     outputImage[row * imageWidth + col] = sum;
//
// }
