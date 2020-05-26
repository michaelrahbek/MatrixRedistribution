# Matrix redistribution

This folder contains a redistribution function using Message Passing Interface with blocking send/receive, non- blocking send/receive, and collective communications (*see subfolders*). For each subfolder the same redistribution is performed where elements of a square $N \times N$-matrix, from one 2D block-cyclic distribution on $p$ processors to another one is performed. Two exacutables are created; one to test correctness *test*, and one for timing to measure the performance *project*.

The initial matrix is distributed such that every element is in a block consisting of $NB_r \times NB_c$ number of elements, where:

- $NB_r$ is the blocksize in the row-dimension,.
- $NB_c$ is the blocksize in the column-dimension.

Each block is assigned a rank, and the number of different ranks are constrained by the size number of processors. The latter is given by $P_r \times P_c$, representing a grid of size number of different ranks, where:

- $P_r$ is the number of processes in the row-dimension.
- $P_c$ is the number of processes in the column-dimension.

The redistribution function redistributes the elements due to a changing of the variables $NB_r$, $NB_c$, $P_r$, and $P_c$. This change requires an element redistribution from its current processor to another.

## Reference

Part of the code is inspired by material handed out from the DTU course *02616 Large-scale Modelling* (https://kurser.dtu.dk/course/02616).



