#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <core/sTimer.h>
#include <core/sRandom.h>
#include <core/helper_math.h>
#include <image/svpng.inc>

// stream compaction
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define PI 3.14159265359f

#define CUDA_SAFE_CALL( call) {										 \
cudaError err = call;                                                    \
if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
            __FILE__, __LINE__, cudaGetErrorString( err) );              \
    getchar();                                                 \
} }

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
    __device__ Ray()
    {
        //active = true;
        /*depth = */pixelIndex = 0;
        origin = direction = make_float3(0.0f, 0.0f, 0.0f);
        //throughput = make_float3(1.0f, 1.0f, 1.0f);
        //L = make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ Ray(float3 origin, float3 direction)
        : origin(origin), direction(direction)
    {
        //active = true;
        /*depth = */pixelIndex = 0;
        //throughput = make_float3(1.0f, 1.0f, 1.0f);
        //L = make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 origin;
    float3 direction;

    //int depth, pixelIndex;
    //float3 L, throughput;
    int pixelIndex;
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
    {1e5f,{50.0f, 40.8f, -1e5f + 170.0f},{0.0f, 0.0f, 0.0f},{0.0f, 0.0f, 0.0f}, DIFFUSE}, //Frnt 
    {1e5f,{50.0f, 1e5f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Botm 
    {1e5f,{50.0f, -1e5f + 81.6f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Top 
    {16.5f,{27.0f, 16.5f, 47.0f},{0.0f, 0.0f, 0.0f},{1, 1, 1}, MIRROR},//Mirr
    {16.5f,{73.0f, 16.5f, 78.0f},{0.0f, 0.0f, 0.0f},{1, 1, 1}, GLASS},//Glas
    {600.0f,{50.0f, 681.6f - .27f, 81.6f},{12, 12, 12},{0.0f, 0.0f, 0.0f}, DIFFUSE}  // Light
};

__device__ float rgbToLuminance(const float3& rgb)
{
    const float YWeight[3] = {0.212671f, 0.715160f, 0.072169f};
    return YWeight[0] * rgb.x + YWeight[1] * rgb.y + YWeight[2] * rgb.z;
}

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

__device__ float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }

__device__ float gammaCorrection(float x)
{
    return pow(clamp(x), 1 / 2.2f);
}

__global__ void initRayPools(int width, int height, int spp, Ray* rays)
{
    // position of current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // index of current pixel
    int i = (height - y - 1) * width + x;

    curandState rs;
    curand_init(i, 0, 0, &rs);

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir 
    float3 cx = make_float3(width * 0.5135f / height, 0.0f, 0.0f);
    // .5135 is field of view angle
    float3 cy = normalize(cross(cx, cam.direction)) * 0.5135f;
    float3 color = make_float3(0.0f);

    //for(int sy = 0; sy < 2; sy++)
    //{
        //for(int sx = 0; sx < 2; sx++)
        //{
    for(int s = 0; s < spp; s++)
    {
        float r1 = curand_uniform(&rs);
        float dx = r1 < 1 ? sqrtf(r1) - 1 : 1 - sqrtf(2 - r1);
        float r2 = curand_uniform(&rs);
        float dy = r2 < 1 ? sqrtf(r2) - 1 : 1 - sqrtf(2 - r2);
        //--! could be super sampling
        float3 d = cam.direction + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);

        // initialize the current
        rays[i * spp + s] = Ray(cam.origin + d * 140, normalize(d));
        rays[i * spp + s].pixelIndex = i;
    }
    //}
//}
}

__global__ void rayStep(Ray* rays, float3* outputGPU)
{
    /*    float t;
        int id = 0;
        // find the ray
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        Ray& r = rays[index];

        // random generator
        curandState rs;
        curand_init(index, 0, 0, &rs);

        // find closest intersection with object's index
        if(!intersectScene(r, t, id))
        {
            // dead ray
            r.depth = -1;

            // update result
            // --! could be replaced by weighted mean
            outputGPU[r.pixelIndex] += r.L;
        }

        const sphere &obj = spheres[id];
        float3 hitpoint = r.origin + r.direction * t;
        float3 normal = normalize(hitpoint - obj.center);
        // front facing normal
        float3 nl = dot(normal, r.direction) < 0 ? normal : normal * -1;

        // prevent self-intersection
        r.origin = hitpoint + nl * 0.05f;

        // add emission
        r.L += r.throughput * obj.emission;

        // different material
        if(obj.type == DIFFUSE)
        {
            // uniform sampling hemisphere
            float r1 = 2 * PI * curand_uniform(&rs);
            float r2 = curand_uniform(&rs);
            float r2s = sqrtf(r2);

            // compute local coordinate on the hit point
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // local to world convert
            r.direction = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
            //pdf = 1.0f / PI;

            // importance sampling no need costheta
            //throughput *= obj.reflectance * dot(r.direction, nl);
            r.throughput *= obj.reflectance;
        }
        else if(obj.type == MIRROR)
        {
            r.direction = r.direction - normal * 2 * dot(normal, r.direction);
            r.throughput *= obj.reflectance;
            //pdf = 1.0f;
        }
        else
        {
            r.origin = hitpoint;

            // Ideal dielectric REFRACTION
            float3 reflectDir = r.direction - normal * 2 * dot(normal, r.direction);
            // Ray from outside going in?
            bool into = dot(normal, nl) > 0;
            float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.direction, nl), cos2t;

            // total internal reflection
            if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)
            {
                r.direction = reflectDir;
                r.throughput *= obj.reflectance;
            }
            else
            {
                // refract or reflect
                float3 tdir = normalize(r.direction*nnt - normal*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t))));

                float a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : dot(tdir, normal));

                float Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);

                if(curand_uniform(&rs) < P)
                {
                    // reflect
                    r.direction = reflectDir;
                    r.throughput *= obj.reflectance * RP;
                }
                else
                {
                    //refract
                    r.direction = tdir;
                    r.throughput *= obj.reflectance * TP;
                    //throughput *= make_float3(1, 0, 0);
                }
            }
        }

        // Russian roulette Stop with at least some probability to avoid getting stuck
        if(r.depth++ >= 5)
        {
            float q = min(0.95f, rgbToLuminance(r.throughput));
            if(curand_uniform(&rs) >= q)
            {
                // dead ray
                r.depth = -1;

                // update result
                // --! could be replaced by weighted mean
                outputGPU[r.pixelIndex] += r.L;
            }
            r.throughput /= q;
        }
        */
    
    // find the ray
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    Ray& r = rays[index];
    float3 L = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    int depth = 0;

    // random generator
    curandState rs;
    curand_init(index, 0, 0, &rs);

    // ray bounce loop
    while(1)
    {
        float t;
        int id = 0;

        // find closest intersection with object's index
        if(!intersectScene(r, t, id))
            break;

        const sphere &obj = spheres[id];
        float3 hitpoint = r.origin + r.direction * t;
        float3 normal = normalize(hitpoint - obj.center);
        // front facing normal
        float3 nl = dot(normal, r.direction) < 0 ? normal : normal * -1; 

        // prevent self-intersection                                                                 
        r.origin = hitpoint + nl * 0.05f;

        //float pdf = 1.0f;

        // add emission
        L += throughput * obj.emission;

        // different material
        if(obj.type == DIFFUSE)
        {
            // uniform sampling hemisphere
            float r1 = 2 * PI * curand_uniform(&rs);
            float r2 = curand_uniform(&rs);
            float r2s = sqrtf(r2);

            // compute local coordinate on the hit point
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // local to world convert
            r.direction = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
            //pdf = 1.0f / PI;

            // importance sampling no need costheta
            //throughput *= obj.reflectance * dot(r.direction, nl);
            throughput *= obj.reflectance;
        }
        else if(obj.type == MIRROR)
        {
            r.direction = r.direction - normal * 2 * dot(normal, r.direction);
            throughput *= obj.reflectance;
            //pdf = 1.0f;
        }
        else
        {
            r.origin = hitpoint;

            // Ideal dielectric REFRACTION
            float3 reflectDir = r.direction - normal * 2 * dot(normal, r.direction);
            // Ray from outside going in?
            bool into = dot(normal, nl) > 0;
            float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.direction, nl), cos2t;

            // total internal reflection
            if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)
            {
                r.direction = reflectDir;
                throughput *= obj.reflectance;
            }
            else
            {
                // refract or reflect
                float3 tdir = normalize(r.direction*nnt - normal*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t))));

                float a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : dot(tdir, normal));

                float Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);

                if(curand_uniform(&rs) < P)
                {
                    // reflect
                    r.direction = reflectDir;
                    throughput *= obj.reflectance * RP;
                }
                else
                {
                    //refract
                    r.direction = tdir;
                    throughput *= obj.reflectance * TP;
                    //throughput *= make_float3(1, 0, 0);
                }
            }
        }

        // Russian roulette Stop with at least some probability to avoid getting stuck
        if(depth++ >= 5)
        {
            float q = min(0.95f, rgbToLuminance(throughput));
            if(curand_uniform(&rs) >= q)
                break;
            throughput /= q;
        }
    }

    float3& output = outputGPU[r.pixelIndex];
    atomicAdd(&output.x, L.x);
    atomicAdd(&output.y, L.y);
    atomicAdd(&output.z, L.z);
    //outputGPU[r.pixelIndex] += L;
}

__global__ void finalAverage(int width, int height, int spp, float3* output)
{
    // position of current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // index of current pixel
    int i = (height - y - 1) * width + x;

    float3 temp = output[i] / spp;
    output[i] = make_float3(clamp(temp.x, 0.0f, 1.0f), clamp(temp.y, 0.0f, 1.0f), clamp(temp.z, 0.0f, 1.0f));
}

//struct isInactive
//{
//    __host__ __device__ bool operator()(const Ray & r)
//    {
//        return (r.depth == -1);
//    }
//};

// -----------------------------------CPU Func-----------------------------------
int toInt(float x)
{
    return (int) (pow(clamp(x, 0.0f, 1.0f), 1.0f / 2.2f) * 255 + 0.5f);
}

void save(const char* fileName, int width, int height, float3* data)
{
    FILE *fp = fopen(fileName, "wb");

    // Convert from float3 array to uchar array
    unsigned char* output = new unsigned char[width * height * 3];

    for(int i = 0; i < width * height; i++)
    {
        //printf_s("%f %f %f \n", data[i].x, data[i].y, data[i].z);
        output[i * 3 + 0] = toInt(data[i].x);
        output[i * 3 + 1] = toInt(data[i].y);
        output[i * 3 + 2] = toInt(data[i].z);
    }

    svpng(fp, width, height, output, 0);
    fclose(fp);
    delete[] output;

    printf("Image %s saved successfully\n", fileName);
}

void devicePropertyPrint()
{
    // Device
    int dev = 0;
    cudaDeviceProp devProp;
    if(cudaGetDeviceProperties(&devProp, dev) == cudaSuccess)
    {
        std::cout << "Device " << dev << ", named: " << devProp.name << std::endl;
        std::cout << "Multi Processor Count£º" << devProp.multiProcessorCount << std::endl;
        std::cout << "Size of SharedMem Per-Block£º" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Max Threads Per-Block£º" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Per-MultiProcessor£º" << devProp.maxThreadsPerMultiProcessor << std::endl;
    }
}

int main(int argc, char *argv[])
{
    devicePropertyPrint();

    // Image Size
    int width = 512, height = 512;
    int spp = argc == 2 ? atoi(argv[1]) : 200;

    printf("\nRendering Size: [%d, %d], spp: %d\n", width, height, spp);
    printf("-----------------------------------Rendering Started-----------------------------------\n");

    sTimer t;

    // Memory allocation
    int pixelCount = width * height;
    float3* outputCPU = new float3[pixelCount];
    float3* outputGPU;
    CUDA_SAFE_CALL(cudaMalloc(&outputGPU, pixelCount * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemset(outputGPU, 0, pixelCount * sizeof(float3)));

    // Ray Pool
    int rayCount = width * height * spp;
    Ray* rays = NULL;
    printf("Mem Cost of Rays: %.1fMB\n", rayCount * sizeof(Ray) / 1048576.0f);
    CUDA_SAFE_CALL(cudaMalloc((void**) &rays, rayCount * sizeof(Ray)));

    // -----------------------------------Rendering Start-----------------------------------
    t.start();

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);

    // init ray pools
    initRayPools<<<gridSize, blockSize >>>(width, height, spp, rays);

    int activeRayCount = rayCount;
    //int depth = 0;
    printf("Initial Ray Counts: %d\n", activeRayCount);
    //while(activeRayCount > 0 && depth < 20)
    //{
        // ray iteration kernel
    unsigned int threadCount = 256;
    unsigned int blockCount = rayCount / threadCount;
    rayStep <<<blockCount, threadCount>>> (rays, outputGPU);

    // stream compaction to reduce the number of rays
    //thrust::device_ptr<Ray> raysStart(rays);
    //thrust::device_ptr<Ray> raysEnd = thrust::remove_if(raysStart, raysStart + activeRayCount, isInactive());
    //activeRayCount = (int)(raysEnd - raysStart);
    //printf("Active Ray Counts: %d\n", activeRayCount);

    //depth++;
//}

// average the color
    finalAverage << <gridSize, blockSize >> > (width, height, spp, outputGPU);

    cudaDeviceSynchronize();
    t.end();
    // -----------------------------------Rendering End-----------------------------------

    // Copy Mem from GPU to CPU
    cudaMemcpy(outputCPU, outputGPU, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(outputGPU);

    printf("-----------------------------------Rendering Ended-----------------------------------\n");

    printf("Cost time: %f\n", t.difference());

    save("test.png", width, height, outputCPU);

    getchar();
    return 0;
}