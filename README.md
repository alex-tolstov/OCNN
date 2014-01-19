OCNN
====

A project for CUDA implementation of Oscillarory Chaotic Neural Network

Folder 'java' is for test of Voronoi diagram computing by Fortune's algorithm.

Folder 'cpp' contains Microsoft Visual C++ Project. The idea of the project is 
to implement OCNN for CUDA videocards, to make visualisation of results, 
to provide usage of FCPS clustering database.

Also it uses binaries of qhull library to compute Delaunay alrogithm in O(N log N).

There are some implementations - using CUDA or using CPU (speedy version, using OpenMP and
vectorization techniques if compiled by Intel Composer XE). CUDA is 8x times faster on
comparable devices - GTX 580 vs Intel i7 920.