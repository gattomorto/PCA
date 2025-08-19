#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct 
{
	float *p;
	int num_righe; //in totale 
	int front;//in numero riga
	int back;// la prima riga libera
	int n;
}coda_t;

extern "C" void azzera_coda(coda_t *c_d, float* pivot_row_d, int j);
extern "C" void initialize_queue(coda_t* c, int n, int num_righe);
extern "C" float *enqueue(coda_t *c);
extern "C" float *dequeue(coda_t *c);
extern "C" void free_queue(coda_t c);
extern "C" void assign_me_to_a_gpu(int rank);
extern "C" void update_matrix(float *A_h, float *row_d, int n);
extern "C" void cudamalloc(int n, float **ptr);
extern "C" void cudamalloc_pinned(int n, float **ptr);
extern "C" void cudafree(float *ptr);
extern "C" void initialize_update_matrix_stream();
extern "C" void synchronize_update_matrix_stream();

const int THs_PER_BLOCK = 128;
cudaStream_t update_matrix_stream;

extern "C" void initialize_update_matrix_stream()
{
	cudaStreamCreate(&update_matrix_stream);
}
extern "C" void synchronize_update_matrix_stream()
{
	cudaStreamSynchronize(update_matrix_stream);
}

//coda: non parte dalla prima cella, ma è gia sfasato correttemente
//pivot_row: è il primo byte senza nessun sfasamento
__global__ void azzera_coda_ker(float* pivot_row, float *coda, int n, int num_rows_in_queue, int ths_per_row, int j)
{
	int th_id = blockIdx.x * blockDim.x + threadIdx.x;
	int th_row = th_id / ths_per_row;
	int th_mem_id = th_id + (th_row+1) * j;
	int last_th_id_in_queue = num_rows_in_queue * ths_per_row - 1;
	float m;

	if(th_id <= last_th_id_in_queue)
	{
		m = coda[th_row*n+j] / pivot_row[j];//1op, 2acc
		coda[th_mem_id] = coda[th_mem_id] - pivot_row[th_mem_id%n] * m;//2op, 3acc

		if(th_id % ths_per_row == 0)
			coda[th_mem_id] = m;//1acc

	}
	
	//CGMA = 3/6= 0.5
}

extern "C" void azzera_coda(coda_t *c_d, float* pivot_row_d, int j)
{
	int n = c_d->n;
	float *coda = c_d->p + c_d->front*n;
	int num_rows_in_queue = c_d->back - c_d->front;
	int num_ths = (n-j) * num_rows_in_queue;
	int num_blocks = num_ths/THs_PER_BLOCK + 1;
	int ths_per_row = n - j;

	azzera_coda_ker <<<num_blocks, THs_PER_BLOCK>>> (pivot_row_d, coda, n,num_rows_in_queue, ths_per_row, j);
}

extern "C" void initialize_queue(coda_t* c, int n,int num_righe)
{
	cudaMalloc((void**) &c->p, n*num_righe*sizeof(float));
	c->num_righe = num_righe;
	c->front = 0;
	c->back = 0;
	c->n = n;
}

extern "C" float *enqueue(coda_t *c)
{
	float *start = c->p+c->back*c->n;
	c->back++;
	return start;
}

//restituisce un indirizzo gpu
extern "C" float *dequeue(coda_t *c)
{
	float *output = c->p+c->front*c->n;
	c->front++;
	return output;
}
extern "C" void free_queue(coda_t c)
{
	cudaFree(c.p);
}
extern "C" void assign_me_to_a_gpu(int rank)
{
	cudaSetDevice(rank);
}

extern "C" void update_matrix(float *A_h, float *row_d, int n)
{
	cudaMemcpyAsync(A_h, row_d,n*sizeof(float), cudaMemcpyDeviceToHost, update_matrix_stream);
}

extern "C" void cudamalloc(int n, float **ptr)
{
	cudaMalloc((void**)ptr, n*sizeof(float));
}
extern "C" void cudamalloc_pinned(int n, float **ptr)
{
	cudaMallocHost((void**)ptr, n*sizeof(float));
}

extern "C" void cudafree(float *ptr)
{
	cudaFree(ptr);
}
extern "C" void cudafree_pinned(float *ptr)
{
	cudaFreeHost(ptr);
}