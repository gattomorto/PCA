#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef struct 
{
	float *p;
	int num_righe;
	int front;//in numero riga
	int back;// la prima riga libera
	int n;
}coda_t;

float azzera_coda(coda_t *c_d, float* pivot_row_d, int j);
void initialize_queue(coda_t* c, int n, int num_righe);
float *enqueue(coda_t *c);
float *dequeue(coda_t *c);
void free_queue(coda_t c);
void assign_me_to_a_gpu(int rank);
void update_matrix(float *A_h, float *row_d, int n);
void cudamalloc(int n, float **ptr);
void cudamalloc_pinned(int n, float **ptr);
void cudafree(float *ptr);
void cudafree_pinned(float *ptr);
void initialize_update_matrix_stream();
void synchronize_update_matrix_stream();

float* generate_strictly_diagonally_dominant_matrix(int n, int seed)
{
	srand(seed);
	float *A = malloc(n * n * sizeof(float));
	for(int i=0; i<n; i++)
	{
		int somma_riga_i=0;
		for(int j=0; j<n; j++)
		{
			int r = rand()%20;
			somma_riga_i+=r;
			A[i*n+j]=r*(somma_riga_i%2==0?1:-1);
		}

		int a_ii = somma_riga_i + rand()%5;
		a_ii*=(a_ii%2==0?1:-1);
		A[i*n+i]=a_ii;
	}
	return A;
}
void my_gemm(float *A,float *LU,float *C, int n)
{
	int ii,jj,kk;
	float sum;
	for (ii = 0; ii < n; ii++){
		for (jj = 0; jj < n ; jj++){
			sum = 0.0;
			for(kk = 0; kk < n; kk++){
				sum += A[ii*n+kk] * LU[kk*n+jj];
			}
			C[ii*n + jj] = sum;
		}
	}
}
float* prendi_L(float *A, int n)
{
	float *L = malloc(sizeof(float)*n*n);
	memcpy(L,A,n*n*sizeof(float));

	for(int i=0; i<n; i++)
	{
		L[i*n+i]=1;
		for(int j = i+1;j<n; j++)
			L[i*n+j]=0;
	}
	return L;
}
float* prendi_U(float *A, int n)
{
	float *U = malloc(sizeof(float)*n*n);
	memcpy(U,A,n*n*sizeof(float));

	for(int j=0;j <n; j++)
		for(int i =j+1;i<n; i++)
			U[i*n+j]=0;
		
	
	return U;
}
float max_err(float *A, float *LxU, int n)
{
	float max_err = abs(A[0]-LxU[0]);
	for(int k=1; k<n*n; k++)
	{
		float err = abs(A[k]-LxU[k]);
		if(err>max_err)
			max_err=err;
	}

	return max_err;
}
void serial_lu_decomposition(float *A, int n)
{
	for(int i =0; i<n-1;i++)
	{
		for(int j=i+1; j<n; j++)
		{
			float m = A[j*n +i]/A[i*n+i];

			for(int k=i+1; k<n; k++)
			{
				A[j*n +k]-=m*A[i*n+k];
			}

			A[j*n + i]=m;
		}
	}
}
int main (int argc, char *argv[])
{
	int my_id, n_proc;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	MPI_Request req;

    int n = 2000;
	float *A,*LU;
    float max_e;
    float *ptr_pivot_row_d;
    coda_t c_d;

    assign_me_to_a_gpu(my_id);
 
    cudamalloc(n, &ptr_pivot_row_d);
    
    if(my_id==0)
    {
        A = generate_strictly_diagonally_dominant_matrix(n,0);
        initialize_update_matrix_stream();
        cudamalloc_pinned(n*n,&LU);
    }
    
    initialize_queue(&c_d, n, n/n_proc+1);

    //PASSO 1
    for(int i=0; i<n; i++)
    {
        if(my_id==0)	
            MPI_Isend(A+n*i, n, MPI_FLOAT, i%n_proc, 0, MPI_COMM_WORLD, &req);
        
        if(my_id==i%n_proc)
            MPI_Recv(enqueue(&c_d), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    }

    for(int j=0; j<n; j++)
    {
        //PASSO 2
        if (my_id == j % n_proc)
            ptr_pivot_row_d = dequeue(&c_d);
        //PASSO 2
        MPI_Bcast(ptr_pivot_row_d, n, MPI_FLOAT, j%n_proc, MPI_COMM_WORLD);

        //PASSO 3
        if (my_id==0)
            update_matrix(LU+n*j, ptr_pivot_row_d, n);
        
        //PASSO 4
        if(j!=n-1)
            azzera_coda(&c_d, ptr_pivot_row_d, j);
    }
        
    cudafree(c_d.p);
    cudafree(ptr_pivot_row_d);

    if(my_id==0)
    {
        synchronize_update_matrix_stream();
        float *serial_LU = malloc(n*n*sizeof(float));
        memcpy(serial_LU,A,n*n*sizeof(float));
        serial_lu_decomposition(serial_LU,n);
        max_e = max_err(LU,serial_LU,n);
        free(serial_LU);

        free(A);
        cudafree_pinned(LU);
       
    }
    
    if(my_id==0)
    {
        fprintf(stderr,"n:%d\tmax err:%f\n",n,max_e);
    }

	MPI_Finalize();

	return 0;
}
