#pragma once

// Thread Safe Random Generator
#include <random>
#include <time.h>
#include <thread>

// --!See https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers for more detail
#if (_MSC_VER <= 1800)
#define thread_local __declspec( thread )
#elif defined (__GCC__)
#define thread_local __thread
#endif

/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
__device__ float randomFloat(const float min = 0.0f, const float max = 1.0f)
{
    static thread_local std::mt19937* generator = nullptr;

    if(!generator)
    {
        std::hash<std::thread::id> hasher;
        generator = new std::mt19937(clock() + hasher(std::this_thread::get_id()));
    }
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(*generator);
}