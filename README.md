# smallptCuda 
A simple cuda version of [smallpt](http://www.kevinbeason.com/smallpt/)  

|   | GTX1080Ti   | Intel Xeon E5 (6C12T) 2.80GHz   | 
| :----- | :-----:  | :----: |
| Resolution| 1024*768 | 1024*768 |
| SPP | 5000 | 5000|
|  Cost Time  | 4.3s |   32min    |

|   | GTX750   | Intel Xeon E5-2665 2.40GHz |
| :-----: | :-----:  | :----: |
| Resolution| 768*768 | 768*768 |
| spp | 2048 | 2048 |
|  Cost Time  | 9.5s |   7.2min    |

 ![result](test.png)  
## Usage 
### Linux 
    $ cd src
    $ make
    $ ./smallpt 5000
### Windows 
enter the folder "smallptCuda"  
open "smallptCuda.sln" with Visual Studio  
compile & run  
