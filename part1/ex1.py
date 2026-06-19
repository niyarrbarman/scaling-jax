"""
Question 1 [int8 matmul]: Say we want to do the matmul X[B,D]⋅DY[D,F]→Z[B,F] in int8 precision (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can do matmuls faster in lower precision.

    How many bytes need to be loaded from memory? How many need to be written back to memory?
    How many total OPs are performed?
    What is the arithmetic intensity?
    What is a roofline estimate for Tmath and Tcomms ? What are reasonable upper and lower bounds for the runtime of the whole operation?
"""

import numpy as np
