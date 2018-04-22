# smallptCuda 
A simple cuda version of [smallpt](http://www.kevinbeason.com/smallpt/) with some memory optimization.

> See more optimization details on [Shawnlu](http://page.shawnlu.ml/post/a-cuda-version-of-smallpt/) and [Bingo](http://bentleyblanks.github.io/)'s blog.

 ![result](test.png) 

## Benchmark
|   | GTX1080Ti   | Intel Xeon E5 (6C12T) 2.80GHz   | 
| :-----: | :-----:  | :----: |
| Resolution| 1024*768 | 1024*768 |
| SPP | 5000 | 5000|
|  Cost Time  | 4.3s |   32.1min    |


|   | GTX750   | Intel Xeon E5 (8C16T) 2.40GHz |
| :-----: | :-----:  | :----: |
| Resolution| 768*768 | 768*768 |
| SPP | 2048 | 2048 |
|  Cost Time  | 19.0s |   7.2min    |
 
## Usage 
### Linux 
    $ git clone https://github.com/BentleyBlanks/smallptCuda.git 
    $ cd smallptCuda 
    $ git checkout release 
    $ cd src && make 
    $ ./smallpt 5000 
    $ display test.png 
### Windows 
Supporting Visual Studio 2015 + Cuda 9.1

> Enter the folder "smallptCuda", open smallptCuda.sln for more details.

