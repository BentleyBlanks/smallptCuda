#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

#include <t3Timer.h>
#include <svpng.inc>
#include <cutil_math.h>
//#include <a3Random.h>

__device__ static float getRandom(unsigned int *seed0, unsigned int *seed1)
{
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union
    {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

// -----------------------------------GPU Func-----------------------------------
__global__ void radiance()
{

}

__global__ void render(int spp, int width, int height, float3* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // seeds for random number generator    
    unsigned int s1 = x;
    unsigned int s2 = y;
    float3 color = make_float3(1.0f);

    // index of current pixel
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //unsigned int i = (height - y - 1) * width + x; // index of current pixel (calculated using thread index) 

    for(int s = 0; s < spp; s++)
    {
        color = make_float3(getRandom(&s1, &s2), getRandom(&s1, &s2), getRandom(&s1, &s2));
    }
    
    // clamp value to [0, 1] and write to GPU Mem
    output[i] = make_float3(clamp(color.x, 0.0f, 1.0f), clamp(color.y, 0.0f, 1.0f), clamp(color.z, 0.0f, 1.0f));
}

// -----------------------------------CPU Func-----------------------------------
void save(const char* fileName, int width, int height, float3* data)
{
    FILE *fp = fopen(fileName, "wb");

    // Convert from float3 array to uchar array
    unsigned char* output = new unsigned char[width * height * 3];

    for(int i = 0; i < width * height; i++)
    {
        //printf_s("%f %f %f \n", data[i].x, data[i].y, data[i].z);
        output[i * 3 + 0] = data[i].x * 255;
        output[i * 3 + 1] = data[i].y * 255;
        output[i * 3 + 2] = data[i].z * 255;
    }

    svpng(fp, width, height, output, 0);
    fclose(fp);
    delete[] output;
}

int main(void)
{
    // Image Size
    int width = 256, height = 256;
    int spp = 1;

    t3Timer t;
    
    // Memory on CPU
    float3* outputCPU = new float3[width * height];
    float3* outputGPU;
    cudaMalloc(&outputGPU, width * height * sizeof(float3));

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);

    t.start();

    // Render on GPU
    render <<<gridSize, blockSize>>>(spp, width, height, outputGPU);

    //cudaDeviceSynchronize();
    t.end();

    // Copy Mem from GPU to CPU
    cudaMemcpy(outputCPU, outputGPU, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(outputGPU);

    printf("Cost time: %f\n", t.difference());

    save("test.png", width, height, outputCPU);

    getchar();
    return 0;
}