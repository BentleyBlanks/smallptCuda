#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

#include <core/sTimer.h>
#include <core/sRandom.h>
#include <core/helper_math.h>
#include <image/svpng.inc>

#define PI 3.14159265359f

// -----------------------------------GPU Func-----------------------------------
// From [smallpt](http://www.kevinbeason.com/smallpt/)
enum materialType
{ 
    DIFFUSE = 0, 
    MIRROR, 
    GLASS
};

struct Ray
{
    __device__ Ray(float3 origin, float3 direction) 
        : origin(origin), direction(direction) {}

    float3 origin;
    float3 direction;
};

struct sphere
{
    float radius;
    float3 center, emission, reflectance;
    materialType type;

    __device__ double intersect(const Ray &r) const
    {

        float3 op = center - r.origin;
        float t, epsilon = 0.0001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.direction);    // b in quadratic equation
        float disc = b*b - dot(op, op) + radius*radius;  // discriminant quadratic equation
        if(disc < 0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
        else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // pick closest point in front of ray origin
    }
};

__constant__ sphere spheres[] = {
    {1e5f,{1e5f + 1.0f, 40.8f, 81.6f},{0.0f, 0.0f, 0.0f},{0.75f, 0.25f, 0.25f}, DIFFUSE}, //Left 
    {1e5f,{-1e5f + 99.0f, 40.8f, 81.6f},{0.0f, 0.0f, 0.0f},{.25f, .25f, .75f}, DIFFUSE}, //Rght 
    {1e5f,{50.0f, 40.8f, 1e5f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Back 
    {1e5f,{50.0f, 40.8f, -1e5f + 600.0f},{0.0f, 0.0f, 0.0f},{1.00f, 1.00f, 1.00f}, DIFFUSE}, //Frnt 
    {1e5f,{50.0f, 1e5f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Botm 
    {1e5f,{50.0f, -1e5f + 81.6f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Top 
    {16.5f,{27.0f, 16.5f, 47.0f},{0.0f, 0.0f, 0.0f},{1.0f, 1.0f, 1.0f}, DIFFUSE}, // small sphere 1
    {16.5f,{73.0f, 16.5f, 78.0f},{0.0f, 0.0f, 0.0f},{1.0f, 1.0f, 1.0f}, DIFFUSE}, // small sphere 2
    {600.0f,{50.0f, 681.6f - .77f, 81.6f},{2.0f, 1.8f, 1.6f},{0.0f, 0.0f, 0.0f}, DIFFUSE}  // Light
};

__device__ bool intersectScene(const Ray &r, float &t, int &id)
{
    float n = sizeof(spheres) / sizeof(sphere), d, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
    for(int i = int(n); i--;)
    {
        // find closest hit object and point
        if((d = spheres[i].intersect(r)) && d < t)
        {
            t = d;
            id = i;
        }
    }
        
    return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

__device__ float gammaCorrection(float x)
{
    return pow(x, 1 / 2.2f);
}

__device__ float3 radiance(Ray &r, curandState* rs)
{
    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    // ray bounce loop (no Russian Roulette used) 
    for(int bounces = 0; bounces < 4; bounces++)
    {  // iteration up to 4 bounces (replaces recursion in CPU code)

        float t;           // distance to closest intersection 
        int id = 0;        // index of closest intersected sphere 

                           // test ray for intersection with scene
        if(!intersectScene(r, t, id))
            return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

                                                  // else, we've got a hit!
                                                  // compute hitpoint and normal
        const sphere &obj = spheres[id];  // hitobject
        float3 x = r.origin + r.direction*t;          // hitpoint 
        float3 n = normalize(x - obj.center);    // normal
        float3 nl = dot(n, r.direction) < 0 ? n : n * -1; // front facing normal

                                                    // add emission of current sphere to accumulated colour
                                                    // (first term in rendering equation sum) 
        accucolor += mask * obj.emission;

        // all spheres in the scene are diffuse
        // diffuse material reflects light uniformly in all directions
        // generate new diffuse ray:
        // origin = hitpoint of previous ray in path
        // random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)

        // create 2 random numbers
        float r1 = 2 * PI * curand_uniform(rs); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
        float r2 = curand_uniform(rs);  // pick random number for elevation
        float r2s = sqrtf(r2);

        // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction 
        // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
        float3 w = nl;
        float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
        float3 v = cross(w, u);

        // compute random ray direction on hemisphere using polar coordinates
        // cosine weighted importance sampling (favours ray directions closer to normal direction)
        float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

        // new ray origin is intersection point of previous ray with scene
        r.origin = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
        r.direction = d;

        mask *= obj.reflectance;    // multiply with colour of object       
        mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
        mask *= 2;          // fudge factor
    }

    return accucolor;
}

__global__ void render(int spp, int width, int height, float3* output)
{
    // position of current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // index of current pixel
    //int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int i = (height - y - 1) * width + x;
    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

    curandState rs;
    curand_init(i, 0, 0, &rs);

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir 
    float3 cx = make_float3(width * 0.5135f / height, 0.0f, 0.0f);
    // .5135 is field of view angle
    float3 cy = normalize(cross(cx, cam.direction)) * 0.5135f;
    float3 color = make_float3(0.0f);

    for(int s = 0; s < spp; s++)
    {
        //--! could be super sampling
        float3 d = cam.direction + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);
        Ray tRay = Ray(cam.origin + d * 40, normalize(d));
        color += radiance(tRay, &rs)*(1. / spp);
    }

    // gamma correction
    color.x = gammaCorrection(color.x);
    color.y = gammaCorrection(color.y);
    color.z = gammaCorrection(color.z);

    // output to the cache
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
    int width = 512, height = 512;
    int spp = 1024;

    sTimer t;
    
    // Memory on CPU
    float3* outputCPU = new float3[width * height];
    float3* outputGPU;
    cudaMalloc(&outputGPU, width * height * sizeof(float3));

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);

    t.start();

    // Render on GPU
    render <<<gridSize, blockSize>>>(spp, width, height, outputGPU);

    cudaDeviceSynchronize();
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