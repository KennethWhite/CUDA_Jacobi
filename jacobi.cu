//Kenneth White Fall 18



// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>
#include <helper_timer.h>


#include "kernel.h"
#include "kernel1.h"


int device = 0;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runCUDA( float *h_dataA, float* h_dataB, int width, int height, int passes, int threadsPerBlock, int shouldPrint);
void runSerial( float * h_dataA, float * h_dataB, int width, int height, int passes, int shouldPrint);
void printArray(float *arr, int rows, int cols, int shouldPrint);
float * serial (float *a1, float*a2, int width, int height, int passes) ;
void initializeArrays(float *a1, float *a2, int width, int height);
void usage();


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

   // jacobi threadsperblock passes width height [p]
   if(argc < 5 ){
   
      usage();
      return 1;
   }
   
   int threadsPerBlock = atoi(argv[1]);
   int passes = atoi(argv[2]);
   int width = atoi(argv[3]);
   int height = atoi(argv[4]);
   int shouldPrint=0;
   
   if(argc == 6 ) {
      if (argv[5][0]=='p'){
         
         shouldPrint=1;
      } else {
      
         usage();
         return 1;
      }
   }
      
   float * h_dataA= (float *)malloc(width * height * sizeof(float));
   float * h_dataB= (float *)malloc(width * height * sizeof(float));

   initializeArrays(h_dataA, h_dataB, width, height);
   
   if (threadsPerBlock == 0){
      
      runSerial(h_dataA, h_dataB, width, height, passes, shouldPrint);
   } else {

      runCUDA(h_dataA, h_dataB, width, height, passes, threadsPerBlock, shouldPrint);
   }
   
   // Clean up Memory
   free( h_dataA);
   free( h_dataB);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the CUDA version
////////////////////////////////////////////////////////////////////////////////
void runCUDA( float *h_dataA, float* h_dataB, int width, int height, int passes, int threadsPerBlock, int shouldPrint){


   // Use card 0  (See top of file to make sure you are using your assigned device.)
   checkCudaErrors(cudaSetDevice(device));

   // To ensure alignment, we'll use the code below to pad rows of the arrays when they are 
   // allocated on the device.
   size_t pitch;
   // allocate device memory for data A
   float* d_dataA;
   checkCudaErrors( cudaMallocPitch( (void**) &d_dataA, &pitch, width * sizeof(float), height));
   
   // copy host memory to device memory for image A
   checkCudaErrors( cudaMemcpy2D( d_dataA, pitch, h_dataA, width * sizeof(float), width * sizeof(float), height,
                             cudaMemcpyHostToDevice) );   
   // repeat for second device array
   float* d_dataB;
   checkCudaErrors( cudaMallocPitch( (void**) &d_dataB, &pitch, width * sizeof(float), height));
   // copy host memory to device memory for image B
   checkCudaErrors( cudaMemcpy2D( d_dataB, pitch, h_dataB, width * sizeof(float), width * sizeof(float), height,
                             cudaMemcpyHostToDevice) );                          
   //***************************
   // setup CUDA execution parameters
   
   int blockHeight;
   int blockWidth;
   
   // When testing with small arrays, this code might be useful. Feel free to change it.
   if (threadsPerBlock > width - 2 ){
   
      blockWidth = 16 * (int) ceil((width - 2) / 16.0); 
      blockHeight = 1;
   } else {
      
      blockWidth = threadsPerBlock;
      blockHeight = 1;
   }
   
   int gridWidth = (int) ceil( (width - 2) / (float) blockWidth);
   int gridHeight = (int) ceil( (height - 2) / (float) blockHeight);
   
   // number of blocks required to process all the data.
   int numBlocks =   gridWidth * gridHeight;
   // Each block gets a shared memory region of this size.
   unsigned int shared_mem_size = ((threadsPerBlock + 2) * 4) * sizeof(float); 
   
   printf("blockDim.x=%d blockDim.y=%d    grid = %d x %d\n", blockWidth, blockHeight, gridWidth, gridHeight);
   printf("numBlocks = %d,  threadsPerBlock = %d   shared_mem_size = %d\n", numBlocks, threadsPerBlock,  shared_mem_size);
   
   if(gridWidth > 65536 || gridHeight > 65536) {
      fprintf(stderr, "****Error: a block dimension is too large.\n");
   }
   
   if(threadsPerBlock > 1024) {
      fprintf(stderr, "****Error: number of threads per block is too large.\n");
   }
   
   if(shared_mem_size > 49152) {
      fprintf(stderr, "****Error: shared memory per block is too large.\n");
   }  
   // Format the grid, which is a collection of blocks. 
   dim3  grid( gridWidth, gridHeight, 1);
   
   // Format the blocks. 
   dim3  threads( blockWidth, blockHeight, 1);
  
   printArray(h_dataA, height, width, shouldPrint);
   StopWatchInterface *timer = NULL;
   sdkCreateTimer(&timer);
   sdkStartTimer(&timer);
     
   float * temp;
   for(int r=0; r<passes; r++){ 
      //execute the kernel
      k1 <<< grid, threads, shared_mem_size >>>( d_dataA, d_dataB, pitch/sizeof(float), width);
      
      //uncomment the following line to use k0, the simple kernel, provided in kernel.cu           
      //k0 <<< grid, threads >>>( d_dataA, d_dataB, pitch/sizeof(float), width);

      // swap the device data pointers  
      temp    = d_dataA;
      d_dataA = d_dataB;
      d_dataB = temp;
   }
   
   // check if kernel execution generated an error
   cudaError_t code = cudaGetLastError();
   if (code != cudaSuccess){
       printf ("Cuda Kernel Launch error -- %s\n", cudaGetErrorString(code));
   }
   cudaThreadSynchronize();
 
   sdkStopTimer(&timer); 
   // copy result from device to host
   checkCudaErrors( cudaMemcpy2D( h_dataA, width * sizeof(float), d_dataA, pitch, width * sizeof(float), height,cudaMemcpyDeviceToHost) );
   
   printArray(h_dataA, height, width, shouldPrint);
   printf( "Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
   sdkDeleteTimer(&timer);

   // cleanup memory
   checkCudaErrors(cudaFree(d_dataA));
   checkCudaErrors(cudaFree(d_dataB));
}

/* Run the serial jacobi code using the referenced arrays of floats with given width and height for
 * the specified number of passes. If the final parameter is non-zero, the initial and final states
 * of the arrays will be printed.  In all cases, the execution time will be printed to stdout.
 *
 * For the first pass, values will be read from h_dataA and written to h_dataB. For subsequent 
 * passes, the role of the arrays will be reversed.
 */
void runSerial( float * h_dataA, float * h_dataB, int width, int height, int passes, int shouldPrint){

   printf("Running Serial Code.\n");
   
   float * serialResult;
   
   printArray(h_dataA, height, width, shouldPrint);
   
   StopWatchInterface *timer = NULL;
   sdkCreateTimer(&timer);
   sdkStartTimer(&timer);


   serialResult = serial(h_dataA, h_dataB, width, height, passes);

   sdkStopTimer(&timer);

   printArray(serialResult, height, width, shouldPrint);
   
   printf( "Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
   sdkDeleteTimer(&timer);

}


/* Performs the specified number of passes of jacobi iteration on two arrays
 * of the given width and height. For the first pass, values will be read from
 * a1 and written to a2. For subsequent passes, the role of the arrays will
 * be exchanged. In all cases, a pointer to the most recently changed array
 * is returned.
 *
 * For each element, this code computes a weighted average of the neighbors
 * and then reduces this value by 5% to simulate heat loss. There is nothing
 * mathematically or physically rigorous about this calculation, and it is
 * simply meant to provide an interesting parallel programming example.
 */
float * serial (float *a1, float*a2, int width, int height, int passes) {
   
   int i,j,p;
   float * old=a1;
   float * New=a2;
   float * temp;
   
   
   for(p=0; p<passes; p++){
      
      
      for(i=1; i<height-1; i++){
         for(j=1; j<width-1; j++){
            
            New[i*width +j] = (
                              0.2f * old[i*width + j] +
            
                              0.1f * old[(i-1) * width +  j   ] +       //N
                              0.1f * old[(i-1) * width + (j+1)] +       //NE
                              0.1f * old[ i    * width + (j+1)] +       //E
                              0.1f * old[(i+1) * width + (j+1)] +       //SE
                              0.1f * old[(i+1) * width +  j   ] +       //S
                              0.1f * old[(i+1) * width + (j-1)] +       //SW
                              0.1f * old[ i    * width + (j-1)] +       //W
                              0.1f * old[(i-1) * width + (j-1)]         //NW
                             ) * 0.95f;
         }
      }
      
      
      temp = New;
      New = old;
      old = temp;
   }
   
   return old;
}


/* Initialize the two arrays referenced by the first two parameters in preparation for 
 * jacobi iteration. The width and height of the arrays are given by the integer parameters.
 * Border elements are set to 5.0 for both arrays, and the interior elements of a1 are
 * set to 1.0.  Interior elements of a2 are not initialized.
 */
void initializeArrays(float *a1, float *a2, int width, int height){

   int i, j;

   for(i=0; i<height; i++){
      for(j=0; j<width; j++){
      
         if(i==0 || j ==0 || i==height-1 || j==width-1){ 
         
            a1[i*width + j] = 5.0;
            a2[i*width + j] = 5.0;
         }else {
         
            a1[i*width + j] = 1.0;
         }
      }
   }
}

/* Print the 2D array of floats referenced by the first parameter. The second and third
 * parameters specify its dimensions, while the last argument indicates whether printing
 * is actually desired at all. No output is produced if shouldPrint == 0.
 */
void printArray(float *arr, int rows, int cols, int shouldPrint){
   if (!shouldPrint)
      return;
          
   int i,j;

   for(i=0; i<rows; i++){
      for(j=0; j<cols; j++){
      
         printf("%04.2f ", arr[i*cols + j]);
      }
      printf("\n");
   }

   printf("\n");
}



/* Prints a short but informative message about program usage.*/
void usage(){

   fprintf(stderr, "usage: jacobi threadsperblock passes width height [p]\n");
   fprintf(stderr, "               (if threadsperblock == 0, serial code is run)\n");
}


