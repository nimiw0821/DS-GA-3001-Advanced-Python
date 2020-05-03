import pyopencl as cl
import numpy as np
from content_based_recommender import preprocess, create_tfidf_matrix

# create the context
context = cl.create_some_context()
# create a command queue in the context
queue = cl.CommandQueue(context)

# create data in numpy arrays
df = preprocess('ml-latest-small/movies.csv', 'genres')
tfidf_matrix = create_tfidf_matrix(df, 'genres')
u_np = tfidf_matrix.toarray()
v_np = u_np.T
sim_mat_np = np.empty((len(u_np), len(u_np))).astype(np.float32)

print(u_np.shape)
print(v_np.shape)
print(sim_mat_np.shape)
print('-' * 20)
print(sim_mat_np[:5])
print('-' * 20)

# create the data buffer
mf = cl.mem_flags
u_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = u_np)
v_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = v_np)
sim_mat_buf = cl.Buffer(context, mf.WRITE_ONLY, sim_mat_np.nbytes)

# create the OpenCL program
program = cl.Program(context, """
__kernel void similarity(__global const float *u, __global const float *v,
                         __global float *sim_mat, int size)
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   sim_mat[i + size * j] = 0;
   for (int k = 0; k < size; k++) {
      sim_mat[i + size * j] += u[k + size * i] * v[j + size * k];
   }
}""").build()

program.similarity(queue, u_np.shape, None, u_buf, v_buf, sim_mat_buf, np.int32(len(u_np[0])))

# Copy the data from the buffer to numpy arrays
cl.enqueue_copy(queue, sim_mat_np, sim_mat_buf)
print(sim_mat_np[:5])
print(sim_mat_np.shape)
print(np.dot(u_np, v_np))
