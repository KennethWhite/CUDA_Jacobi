#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float s_data[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
   extern __shared__ float s_data[];
    // TODO, implement this kernel below
    
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t s_dim_x = blockDim.x + 2;
    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;
    
    // edges of the array
    if (x == 0 || y == 0 || x >= (width - 1)) return;
    
    // if true, left edge of block
    if (tid_x == 0) {      
                         
        s_data[tid_y*s_dim_x + tid_x] = g_dataA[(y - 1) * floatpitch + (x-1)];  // store NW
        s_data[(tid_y+ 1) * s_dim_x + tid_x] = g_dataA[y*floatpitch + (x-1)]; // store W
        s_data[(tid_y+2) * s_dim_x + tid_x] = g_dataA[(y + 1)*floatpitch + (x - 1)];// store SW
    }
    
    // right edge of the block
    if (tid_x == (blockIdx.x - 1) || x <= width - 2) {
    
        s_data[tid_y*s_dim_x + threadIdx.x + 2] = g_dataA[(y - 1) * floatpitch + x + 1];    // NE
        s_data[(tid_y+1) * s_dim_x + threadIdx.x + 2] = g_dataA[y * floatpitch + x + 1];    // E
        s_data[(tid_y+2) * s_dim_x + threadIdx.x + 2] = g_dataA[(y + 1) * floatpitch +x+1]; // SE
    }
    
    // else threads are within the block boundaries
    s_data[(tid_y+2)*s_dim_x+tid_x+1] = g_dataA[(y+1)*floatpitch+x];// N
    s_data[(tid_y+1)*s_dim_x+tid_x+1] = g_dataA[y*floatpitch+x]; // itself
    s_data[tid_y * s_dim_x+tid_x+1] = g_dataA[(y-1)*floatpitch+x];    // S
    __syncthreads();
    
g_dataB[y * floatpitch + x] = (
			0.2f * s_data[(tid_y+1) * s_dim_x + tid_x + 1]  +         //itself
			0.1f * s_data[tid_y*s_dim_x + tid_x + 1]        +         //N
			0.1f * s_data[tid_y*s_dim_x + tid_x + 2]        +         //NE
			0.1f * s_data[(tid_y+1) * s_dim_x + tid_x + 2]  +         //E
			0.1f * s_data[(tid_y+2) * s_dim_x + tid_x + 2]  +         //SE
			0.1f * s_data[(tid_y+2) * s_dim_x + tid_x + 1]  +         //S
			0.1f * s_data[(tid_y+2) * s_dim_x + tid_x]      +         //SW
			0.1f * s_data[(tid_y+1) * s_dim_x + tid_x]      +         //W
			0.1f * s_data[tid_y * s_dim_x + tid_x]                      //NW
		   ) * 0.95f;
    
}


//This version of Kernel uses optimization by copying the data into shared memory and hence results in better performance
//Based upon example kernel at https://developer.nvidia.com/cuda-education
__global__ void calculateCFD_V2( float* g_dataA, float* g_dataB, int floatpitch, int width)
{

	float h = 1.0f/(width-1);

	//Current Global ID
	int i = blockDim.y * blockIdx.y + threadIdx.y; // Y - ID
	int j = blockDim.x * blockIdx.x + threadIdx.x; // X - ID
	
	//Current Local ID (lXX --> refers to local ID i.e. inside a block)
	int block_y = threadIdx.y;
	int block_x = threadIdx.x;
	
	// s_XX --> variables refers to expanded shared memory location in order to accomodate halo elements
	//Current Local ID with 1 offset.
	int s_block_y = block_y + 1;
	int s_block_x = block_x + 1;

	// Variable pointing at top and bottom neighboring location
	int s_block_y_prev = s_block_y - 1;
	int s_block_y_next = s_block_y + 1;

	// Variable pointing at left and right neighboring location
	int s_block_x_prev = s_block_x - 1;
	int s_block_x_next = s_block_x + 1;

	extern __shared__ float s_data[];
	
	unsigned int index = (i)* floatpitch + (j) ;

	if( block_y<1 ) // copy top and bottom halo
	{
		//Copy Top Halo Element
		if(blockIdx.y > 0) // Boundary check
			s_data[block_y*blockDim.x+s_block_x] = g_dataA[index - 1 * floatpitch];

		//Copy Bottom Halo Element
		if(blockIdx.y < (gridDim.y-1)) // Boundary check
			s_data[s_block_y*blockDim.x+blockDim.y+ s_block_x] = g_dataA[index + blockDim.y * floatpitch];
  
	}

	if( block_x<1 ) // copy left and right halo
	{
		if( blockIdx.x > 0) // Boundary check
			s_data[block_y*blockDim.x+s_block_x] = g_dataA[index - 1];
		
		if(blockIdx.x < (gridDim.x-1)) // Boundary check
			s_data[block_y*blockDim.x+s_block_x+blockDim.x] = g_dataA[index + blockDim.x];
	}
	
	// copy current location
	s_data[block_y+s_block_x*blockDim.x] = g_dataA[index]; 

	__syncthreads( );

	if( i > 0 && j > 0 && i < (width-1) && j <(floatpitch-1))
		g_dataB[index] = 0.25f * (s_data[s_block_y_prev*blockDim.x+s_block_x] + s_data[s_block_y_next*blockDim.x+s_block_x] + s_data[s_block_y*blockDim.x + s_block_x_prev] 
			+ s_data[s_block_y*blockDim.x+ s_block_x_next] - 4*h*h);
	
}
