/*
 * Russell Taylor(rtaylor)
 * Matt Crusse(macrusse)
 * CPE458-01 Lab 1 Winter 2013 
 */

#include <sys/stat.h>
#include <sys/mman.h> 
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>

#define TILE_SIZE 32

/*Compile-Time Declaration on double or float usage*/
#ifdef DOUBLE
#define TYPEUSE double

#else
#define TYPEUSE float

#endif

/* 
 * Handles CUDA errors, taking from provided sample code on clupo site
 */

static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



/*Reads Input File and Returns Buffer of Contents*/
char* read_file(const char * file_name) 
{
  size_t size;
  char *buffer;
  FILE *fp;
  
  fp = fopen(file_name,"r");
  if(!fp) {
    fprintf (stderr, "Error opening input file.\n");
    exit (EXIT_FAILURE);    
  }

  fseek (fp, 0, SEEK_END);
  size = ftell(fp);
  rewind (fp);
  
  buffer = (char*) malloc (sizeof(char)*size);
  fread (buffer, 1, size, fp);
  fclose(fp);
  return buffer;
}

/*Calculate the Resultant Matrix from Multiplication*/
void calc_matrix(TYPEUSE *A, TYPEUSE *B, TYPEUSE *C, int Arow, int Acol, int Brow, int Bcol)
{
  uint64_t i, j, k;
  TYPEUSE sum = 0;
  for(i = 0; i < Arow; i++)//Iterate through Matrix B columnwise
  {
    for(j = 0; j < Bcol; j++)//Iterate through Matrix A rowwise
    {
        for(k = 0; k < Acol; k++)//Acol = Brow on valid Matrices
        {
          if(i >475)
            printf("");
          sum+= A[ i* (Acol) + k] * B[k * (Bcol) + j];
          
        }
        C[i *Acol + j] = sum;
        sum = 0;
    }
  }
}

/* Print matrix values to a file outputfile */
void output_matrix(const char * outputfile, TYPEUSE *matrix, int row, int col) 
{
  int i, j;

  FILE *ofp = fopen(outputfile, "w");
  if(!ofp){
    fprintf (stderr, "Error opening output file.\n");
    exit (EXIT_FAILURE);    
  }

  for(i = 0; i < row; i++) {
    for(j = 0; j < col; j++) {
      fprintf(ofp, "%.2f ",matrix[i*uint64_t(col) + j]);
    }  
    if(i < row-1){
      fprintf(ofp, "\n");
    }
  }
  fclose(ofp);
}


/*
 * Simply prints out the matrix to screen 
 */
void print_matrix(TYPEUSE *matrix, int row, int col) 
{
  int i, j;
  for(i = 0; i < row; i++) {
    for(j = 0; j < col; j++) {
      //printf("(%d,%d)", i, j);
      printf("%.2f ",matrix[i*col +j]);
    }  
    if(i < row-1){
      printf("\n");
    }
  }
  printf("\n");

}

/*Created a Matrix based on Buffered Input Information*/
TYPEUSE * read_matrix(int * rowCnt, int * colCnt, char * mapped)
{
  TYPEUSE value;  
  const char *delim_space = " ";
  char *token = NULL;  
  char *unconverted;
  int i, j, len;
  TYPEUSE *matrix;
  uint64_t bigiter;
  *colCnt = 0;
  *rowCnt = 0;

  
  /* Determine Col Count */
  i = 0;
  while(mapped[i] != '\n'){
    if(mapped[i] == '.') {
     (*colCnt)++;
    }
    i++;
  }  

  /* Determine Row Count */
  bigiter = 0;//For large file sizes, an int is too small to iterate through
  len = strlen(mapped);
  while(bigiter < len && mapped[bigiter] != '\0'){
    if((mapped[bigiter] == '\n') && (mapped[bigiter+1] != '\0') ) {
     (*rowCnt)++;
    }
    bigiter+=1;
  }
  (*rowCnt)++;

  /* Malloc the Matrix */
  if (( matrix = (TYPEUSE *) malloc((*rowCnt) * (*colCnt) * sizeof(TYPEUSE))) == NULL ) {
    printf("malloc issue");
  }
    
  /* Read values into matrix */
  i = 0; j = 0;
  for (token = strtok(mapped, delim_space); token != NULL; token = strtok(NULL, delim_space)) {
    value = strtod(token, &unconverted);
    matrix[i*(*colCnt) +j] = value;
    j++;
    if(j == (*colCnt)) {
      j = 0;
      if(++i == (*rowCnt))
	      break;
    }
  }
  return matrix;

}

__device__ void copyMiniMatrix(TYPEUSE * M_device, TYPEUSE M_shared[TILE_SIZE][TILE_SIZE], uint64_t row, uint64_t col)
{
  if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
    M_shared[threadIdx.y][threadIdx.x] = M_device[row + col];  
  }

}

__global__ void MMKernel(TYPEUSE *A_d, TYPEUSE *B_d, TYPEUSE * C_d, uint64_t depth, uint64_t Arow, uint64_t Bcol)
{
  TYPEUSE Cvalue = 0.0;
  __shared__ TYPEUSE A_shared[TILE_SIZE][TILE_SIZE], B_shared[TILE_SIZE][TILE_SIZE]; 
  int resultWidth = Bcol;
  int resultCol = blockIdx.x * blockDim.x + threadIdx.x;
  int resultRow = blockIdx.y * blockDim.y + threadIdx.y;  
  int resultIndex = resultRow * resultWidth + resultCol;

  /* Boundary check */
  if(resultRow >= Arow || resultCol >= Bcol) {
    return;
  }
  
  for(int i = 0; i < (depth+TILE_SIZE-1)/TILE_SIZE; i++)
  {
  
    /* Copy global matricies into shared memory */
    if(threadIdx.x + i* TILE_SIZE < depth){
      copyMiniMatrix(A_d, A_shared, resultRow*Arow, threadIdx.x + (i * TILE_SIZE));
    }
    if(threadIdx.y + i* TILE_SIZE < depth) {
      copyMiniMatrix(B_d, B_shared, (threadIdx.y + (i * TILE_SIZE))*Bcol, resultCol);
    }
    
    /* Wait for all threads to complete copy to shared memory */
    __syncthreads();


    for(int k = 0; k < TILE_SIZE; k++)
    {
      if(k + (i * TILE_SIZE) < depth) /* Boundary check */
      {
        TYPEUSE Aelem = A_shared[threadIdx.y][k];
        TYPEUSE Belem = B_shared[k][threadIdx.x];
        Cvalue += Aelem * Belem;
      }
    }

    /* Wait for all threads to finish */
    __syncthreads();
  }
    
  C_d[resultIndex] = Cvalue;
}

int main (int argc, const char * argv[])
{
  const char * Cfile = "result.out";
  TYPEUSE * Amatrix, * Bmatrix, * Cmatrix;
  TYPEUSE * A_d, * B_d, * C_d;
  int Arow, Acol, Brow, Bcol;
  int size;
  int blockRow, blockCol;
  char * Amapped, * Bmapped;

  if(argc != 3) { 
    fprintf(stderr, "Usage: [Matrix A] [Matrix B]\n");
    exit(EXIT_FAILURE);
  }

  /* Device Properties */
  /*
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  printf("maxThreads: %d\n", prop.maxThreadsPerBlock);
  */

  /* Read and Map matrix */
  Amapped = read_file(argv[1]);
  Bmapped = read_file(argv[2]);
  Amatrix = read_matrix(&Arow, &Acol, Amapped); 
  Bmatrix = read_matrix(&Brow, &Bcol, Bmapped);
  if(Acol != Brow) {
    fprintf(stderr, "Matrices are not a compatible size to be multiplied\n");
    exit(EXIT_FAILURE);
  }
  
  /* Malloc a New Matrix */
  if (( Cmatrix = (TYPEUSE *) malloc((Arow) * (Bcol) * sizeof(TYPEUSE))) == NULL ) {
    printf("malloc issue");
  }
  
  /* Malloc and Copy space on GPU */
  size = Arow * Acol * sizeof(TYPEUSE);
  HANDLE_ERROR(cudaMalloc(&A_d, size));
  HANDLE_ERROR(cudaMemcpy(A_d, Amatrix, size, cudaMemcpyHostToDevice));
  
  size = Brow * Bcol * sizeof(TYPEUSE);
  HANDLE_ERROR(cudaMalloc(&B_d, size));
  HANDLE_ERROR(cudaMemcpy(B_d, Bmatrix, size, cudaMemcpyHostToDevice));

  size = Arow * Bcol * sizeof(TYPEUSE);
  HANDLE_ERROR(cudaMalloc(&C_d, size));
  
  blockRow = (Arow+31) / 32;
  blockCol = (Bcol+31) / 32;
  
  //printf("blockRow: %d\t blockCol: %d\n",blockRow,blockCol);
    
  /*Kernel Call*/
  dim3 dimGrid(blockCol,blockRow);
  dim3 dimBlock(32,32);
  MMKernel<<<dimGrid,dimBlock>>>(A_d, B_d, C_d, Brow, Arow, Bcol);

  HANDLE_ERROR(cudaMemcpy(Cmatrix,C_d,size, cudaMemcpyDeviceToHost));

  //output_matrix(Cfile, Cmatrix, Arow, Bcol);
  
  print_matrix(Cmatrix, Arow, Bcol);
  
  /* Free Stuff */
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(Amatrix);
  free(Bmatrix);
  free(Cmatrix);
  free(Amapped);
  free(Bmapped);

  return 0;
}
