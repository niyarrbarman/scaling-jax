# notes

- model is basically a bunch of matrix multiplications. so a lot of FLoating point addition and multiplication OPerations (FLOPs)
- communication within a chip : within an accelator (TPU/GPU), tensors need to be transferred from HBM (High Bandwidth Memory (80GB on H100)) to compute cores inside the SMs. the bandwidth of this link is called as _HBM bandwidth_
- communication across chips: measured in bytes/s
```
                                    communcation bytes
total communication time  =  ----------------------------------
                              network/memory bandwidth bytes/s
```
- mostly computation within a chip can be overlapped with communication time within a chip and between chips. so we can lower bound training and inference time by taking maximum of computation and comm time and can upper bound with their sum. in practice, the optimization is done against the maximum. so by doing this the lower bound and upper bound differ by at max a factor of 2.
```
T_math + T_comms <= 2 * max(T_math, T_comms)
T_upper = T_math + T_comms
```
- if we assume perfect overlap communication and computation, so when T_math > T_comm, this means we are compute-bound and if T_comm > T_math, we are communication-bound. some FLOPs are always wasted waiting for data (spikes in GPU utilization plots). 
- arithmetic intensity: given by total ratio of FLOPs to the number of bytes it needs to communicate. so _FLOPs per byte_ of a given operation.
```
                         computation FLOPs 
Arithmetic Intensity = ---------------------
                        communication bytes
```
- when T_math is large compared to T_comms, it means we use most of the available FLOPs but when the opposite happens, it means we are spending more time on communication and thus wasting FLOPs. the point of crossover between them is called _peak arithmetic intensity_.
