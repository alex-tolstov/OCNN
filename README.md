OCNN
====

A project for CUDA implementation of Oscillarory Chaotic Neural Network

Current C++ and Java versions are test ones.

You can use it as you wish.

Some known troubles:
1) problem with accuracy - if input data consists of numbers of wide range, it could be a crash or another problem.
2) implemented algorithm is Fortune's one. It has complexity O(N log N), but there is a constant appeared due to creating objects,
heavyweight geometric operations, etc. In future, there will be more optimized version. Now it works ~2 seconds (Java) on my laptop
with Intel Core i3 (Sandy Bridge, mobile).
3) in nearest few days there will be an implementation of chaotic neural network (idea by Zhukova and Benderskaya) using NVIDIA CUDA C,
coupled with Voronoi diagram implementation.


Have a nice day,
Alex Tolstov.