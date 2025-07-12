# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/old/testfaissgpu.py

import faiss
import numpy as np

# Initialize GPU resources
res = faiss.StandardGpuResources()

d = 128  # vector dimension
index_cpu = faiss.IndexFlatL2(d)  # CPU index
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # move index to GPU device 0

print("Created GPU index")

# Create some random vectors and add to index
xb = np.random.random((1000, d)).astype('float32')
index_gpu.add(xb)

print(f"Added {index_gpu.ntotal} vectors to GPU index")
