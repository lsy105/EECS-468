/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

//define tile size
#define TILE_SIZE 32

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

//define helper function
__device__ Matrix GetSubMatrix(Matrix A, int row, int col);

//kernel program
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	
	__shared__ float MTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float NTile[TILE_SIZE][TILE_SIZE + 1];
	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;
	
	int RepeatTimes = M.width/TILE_SIZE + ((M.width%TILE_SIZE) ? 1 : 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
	
		Matrix Msub = GetSubMatrix(M, blockIdx.y, i);
		Matrix Nsub = GetSubMatrix(N, i, blockIdx.x);
		
		if((i * TILE_SIZE + threadIdx.x) < M.width && Row < M.height)
			MTile[threadIdx.y][threadIdx.x] = Msub.elements[threadIdx.y * Msub.pitch + threadIdx.x];
		else
			MTile[threadIdx.y][threadIdx.x] = 0;
		if(Col < N.width && (i * TILE_SIZE + threadIdx.y) < N.height)
			NTile[threadIdx.y][threadIdx.x] = Nsub.elements[threadIdx.y * Nsub.pitch + threadIdx.x];
		else
			NTile[threadIdx.y][threadIdx.x] = 0;
		
		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += MTile[threadIdx.y][j] * NTile[j][threadIdx.x];
		}
		__syncthreads();
	}
	if (Row < P.height && Col < P.width)
		P.elements[Row * P.pitch + Col] = Pvalue;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = TILE_SIZE;
	Asub.height = TILE_SIZE;
	Asub.pitch = A.pitch;
	Asub.elements = &A.elements[A.pitch * TILE_SIZE * row + TILE_SIZE * col];
	return Asub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
