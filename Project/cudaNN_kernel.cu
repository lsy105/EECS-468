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

/* cuda backpropagation neural networks
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

//define tile size
#define TILE_SIZE 32

#include <stdio.h>
#include <math.h>
#include "cudaNN.h"

////////////////////////////////////////////////////////////////////////////////
//! BPNetWork test on kernel for device functionality
////////////////////////////////////////////////////////////////////////////////

//define helper function
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = TILE_SIZE;
	Asub.height = TILE_SIZE;
	Asub.pitch = A.pitch;
	Asub.elements = &A.elements[A.pitch * TILE_SIZE * row + TILE_SIZE * col];
	return Asub;
}

__device__ Matrix GetSubMatrixWithSize(Matrix A, int row, int col, int width, int height)
{
	Matrix Asub;
	Asub.width = width;
	Asub.height = height;
	Asub.pitch = A.pitch;
	Asub.elements = &A.elements[A.pitch * Asub.width * row + Asub.height * col];
	return Asub;
}

//kernel program
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float MTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float NTile[TILE_SIZE][TILE_SIZE + 1];
	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	int RepeatTimes = M.width/TILE_SIZE + (M.width%TILE_SIZE != 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
		Matrix Msub = GetSubMatrix(M, blockIdx.y, i);
		Matrix Nsub = GetSubMatrix(N, i, blockIdx.x);
		
        float inScale_x = ((i * TILE_SIZE + threadIdx.x) < M.width && Row < M.height);
        MTile[threadIdx.y][threadIdx.x] = Msub.elements[threadIdx.y * Msub.pitch + threadIdx.x] * inScale_x;

        float inScale_y = (Col < N.width && (i * TILE_SIZE + threadIdx.y) < N.height);
		NTile[threadIdx.y][threadIdx.x] = Nsub.elements[threadIdx.y * Nsub.pitch + threadIdx.x] * inScale_y;
		
		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += MTile[threadIdx.y][j] * NTile[j][threadIdx.x];
		}

		__syncthreads();
	}
    
	if (Row < P.height && Col < P.width) {
		P.elements[Row * P.pitch + Col] = Pvalue;
    }
}

__global__ void SubMatrixDotMulKernel(Matrix M, Matrix N, Matrix P)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	Matrix Msub = GetSubMatrixWithSize(M, 0, 1, M.width - 1, M.height);
	
	P.elements[Row * P.pitch + Col] = Msub.elements[Row * Msub.pitch + Col] * N.elements[Row * N.pitch + Col];
}

__global__ void MatrixDotMulKernel(Matrix M, Matrix N, Matrix P)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	P.elements[Row * P.pitch + Col] = M.elements[Row * M.pitch + Col] * N.elements[Row * N.pitch + Col];
}

__global__ void MatrixSubKernel(Matrix M, Matrix N, Matrix P)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	P.elements[Row * P.pitch + Col] = M.elements[Row * P.pitch + Col] - N.elements[Row * P.pitch + Col];
}

__global__ void MatrixConstMulKernel(Matrix M, float factor)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	M.elements[Row * M.pitch + Col] = M.elements[Row * M.pitch + Col] * factor;
}

__global__ void MatrixTransposeMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float MTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float NTile[TILE_SIZE][TILE_SIZE + 1];
	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	int RepeatTimes = M.width/TILE_SIZE + (M.width%TILE_SIZE != 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
		Matrix Msub = GetSubMatrix(M, blockIdx.y, i);
		Matrix Nsub = GetSubMatrix(N, blockIdx.y, i);
		
        float inScale_x = ((i * TILE_SIZE + threadIdx.x) < M.width && Row < M.height);
        MTile[threadIdx.y][threadIdx.x] = Msub.elements[threadIdx.y * Msub.pitch + threadIdx.x] * inScale_x;

        float inScale_y = (Col < N.width && (i * TILE_SIZE + threadIdx.y) < N.height);
		NTile[threadIdx.y][threadIdx.x] = Nsub.elements[threadIdx.y * Nsub.pitch + threadIdx.x] * inScale_y;

		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += MTile[j][threadIdx.y] * NTile[j][threadIdx.y];
		}

		__syncthreads();
	}
    
	if (Row < P.height && Col < P.width) {
		P.elements[Row * P.pitch + Col] = Pvalue;
    }
}

__global__ void MatrixMulTransposeKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float MTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float NTile[TILE_SIZE][TILE_SIZE + 1];
	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	int RepeatTimes = M.width/TILE_SIZE + (M.width%TILE_SIZE != 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
		Matrix Msub = GetSubMatrix(M, blockIdx.y, i);
		Matrix Nsub = GetSubMatrix(N, blockIdx.y, i);
		
        float inScale_x = ((i * TILE_SIZE + threadIdx.x) < M.width && Row < M.height);
        MTile[threadIdx.y][threadIdx.x] = Msub.elements[threadIdx.y * Msub.pitch + threadIdx.x] * inScale_x;

        float inScale_y = (Col < N.width && (i * TILE_SIZE + threadIdx.y) < N.height);
		NTile[threadIdx.y][threadIdx.x] = Nsub.elements[threadIdx.y * Nsub.pitch + threadIdx.x] * inScale_y;

		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += MTile[threadIdx.y][j] * NTile[threadIdx.y][j];
		}

		__syncthreads();
	}
    
	if (Row < P.height && Col < P.width) {
		P.elements[Row * P.pitch + Col] = Pvalue;
    }
}

__global__ void MatrixMulTransposePaddingKernel(Matrix X, Matrix a, Matrix Theta)
{
	__shared__ float XTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float ThetaTile[TILE_SIZE][TILE_SIZE + 1];

	Matrix Theta0 = GetSubMatrixWithSize(Theta, 0, 1, Theta.width - 1, Theta.height);

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	int RepeatTimes = X.width/TILE_SIZE + (X.width%TILE_SIZE != 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
		Matrix Xsub = GetSubMatrix(X, blockIdx.y, i);
		Matrix Thetasub = GetSubMatrix(Theta0, blockIdx.y, i);
		
        float inScale_X = ((i * TILE_SIZE + threadIdx.x) < X.width && Row < X.height);
        XTile[threadIdx.y][threadIdx.x] = Xsub.elements[threadIdx.y * Xsub.pitch + threadIdx.x] * inScale_X;

        float inScale_Theta = ((i * TILE_SIZE + threadIdx.x) < Theta0.width && Row < Theta0.height);
		ThetaTile[threadIdx.y][threadIdx.x] = Thetasub.elements[threadIdx.y * Thetasub.pitch + threadIdx.x] * inScale_Theta;

		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += XTile[threadIdx.y][j] * ThetaTile[threadIdx.y][j];
		}

		__syncthreads();
	}
    
	Pvalue += Theta.elements[threadIdx.y * Theta.pitch]; //Padding column

	if (Row < a.height && Col < a.width) {
		a.elements[Row * a.pitch + Col] = Pvalue;
    }
}

__global__ void ForwardPropagateBetweenLayers(Matrix X, Matrix a, Matrix Theta)
{
	__shared__ float XTile[TILE_SIZE][TILE_SIZE + 1];
	__shared__ float ThetaTile[TILE_SIZE][TILE_SIZE + 1];

	Matrix Theta0 = GetSubMatrixWithSize(Theta, 0, 1, Theta.width - 1, Theta.height);

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	int RepeatTimes = X.width/TILE_SIZE + (X.width%TILE_SIZE != 0);
	
	for(int i = 0; i < RepeatTimes; i++) {
		Matrix Xsub = GetSubMatrix(X, blockIdx.y, i);
		Matrix Thetasub = GetSubMatrix(Theta0, blockIdx.y, i);
		
        float inScale_X = ((i * TILE_SIZE + threadIdx.x) < X.width && Row < X.height);
        XTile[threadIdx.y][threadIdx.x] = Xsub.elements[threadIdx.y * Xsub.pitch + threadIdx.x] * inScale_X;

        float inScale_Theta = ((i * TILE_SIZE + threadIdx.x) < Theta0.width && Row < Theta0.height);
		ThetaTile[threadIdx.y][threadIdx.x] = Thetasub.elements[threadIdx.y * Thetasub.pitch + threadIdx.x] * inScale_Theta;

		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; j++) {
			Pvalue += XTile[threadIdx.y][j] * ThetaTile[threadIdx.y][j];
		}

		__syncthreads();
	}
    
	Pvalue += Theta.elements[threadIdx.y * Theta.pitch]; //Padding column

	if (Row < a.height && Col < a.width) {
		a.elements[Row * a.pitch + Col] = 1 / (1 + exp(-Pvalue)); //sigmoid function
    }
}

__global__ void buildVector(Matrix y, Matrix y_vector)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	int index = Col * y_vector.pitch + y.elements[Col]; //Use Col as Row and y.elements[Col] as Column

	y_vector.elements[index] = 1;
}

__global__ void ComputeJ(Matrix J, Matrix y, Matrix a, Matrix Theta1, Matrix Theta2, float lambda)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	J.elements[Row * J.pitch + Col] = ( - y.elements[Row * y.pitch + Col]) * log(a.elements[Row * a.pitch + Col])
                                      - (1 - y.elements[Row * y.pitch + Col]) * log(1 - a.elements[Row * a.pitch + Col]);
}

__global__ void TwoDimSum(Matrix M, float *result)
{
	__shared__ float MTile[TILE_SIZE][TILE_SIZE + 1];

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	MTile[threadIdx.y][threadIdx.x] = M.elements[Row * M.pitch + Col];

	__syncthreads();
	
	if(threadIdx.y < 16) {
		MTile[threadIdx.y][threadIdx.x] += MTile[threadIdx.y + 16][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 8) {
		MTile[threadIdx.y][threadIdx.x] += MTile[threadIdx.y +  8][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 4) {
		MTile[threadIdx.y][threadIdx.x] += MTile[threadIdx.y +  4][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 2) {
		MTile[threadIdx.y][threadIdx.x] += MTile[threadIdx.y +  2][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 1) {
		MTile[threadIdx.y][threadIdx.x] += MTile[threadIdx.y +  1][threadIdx.x];
		__syncthreads();
	}

	if(threadIdx.x < 16 && threadIdx.y == 0) {
		MTile[0][threadIdx.x] += MTile[0][threadIdx.x + 16];
		__syncthreads();
	}
	if(threadIdx.x < 8) {
		MTile[0][threadIdx.x] += MTile[0][threadIdx.x +  8];
		__syncthreads();
	}
	if(threadIdx.x < 4) {
		MTile[0][threadIdx.x] += MTile[0][threadIdx.x +  4];
		__syncthreads();
	}
	if(threadIdx.x < 2) {
		MTile[0][threadIdx.x] += MTile[0][threadIdx.x +  2];
		__syncthreads();
	}
	if(threadIdx.x < 1) {
		MTile[0][threadIdx.x] += MTile[0][threadIdx.x +  1];
		__syncthreads();
	}
	
	atomicAdd(result, MTile[0][0]);
}

__global__ void TwoDimSumTheta(Matrix Theta, float *result)
{
	__shared__ float ThetaTile[TILE_SIZE][TILE_SIZE + 1];

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	Matrix Thetasub = GetSubMatrixWithSize(Theta, 0, 1, Theta.width - 1, Theta.height);
	
	ThetaTile[threadIdx.y][threadIdx.x] = Thetasub.elements[Row * Thetasub.pitch + Col] * Thetasub.elements[Row * Thetasub.pitch + Col];
	
	__syncthreads();
	
	if(threadIdx.y < 16) {
		ThetaTile[threadIdx.y][threadIdx.x] += ThetaTile[threadIdx.y + 16][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 8) {
		ThetaTile[threadIdx.y][threadIdx.x] += ThetaTile[threadIdx.y +  8][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 4) {
		ThetaTile[threadIdx.y][threadIdx.x] += ThetaTile[threadIdx.y +  4][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 2) {
		ThetaTile[threadIdx.y][threadIdx.x] += ThetaTile[threadIdx.y +  2][threadIdx.x];
		__syncthreads();
	}
	if(threadIdx.y < 1) {
		ThetaTile[threadIdx.y][threadIdx.x] += ThetaTile[threadIdx.y +  1][threadIdx.x];
		__syncthreads();
	}

	if(threadIdx.x < 16 && threadIdx.y == 0) {
		ThetaTile[0][threadIdx.x] += ThetaTile[0][threadIdx.x + 16];
		__syncthreads();
	}
	if(threadIdx.x < 8) {
		ThetaTile[0][threadIdx.x] += ThetaTile[0][threadIdx.x +  8];
		__syncthreads();
	}
	if(threadIdx.x < 4) {
		ThetaTile[0][threadIdx.x] += ThetaTile[0][threadIdx.x +  4];
		__syncthreads();
	}
	if(threadIdx.x < 2) {
		ThetaTile[0][threadIdx.x] += ThetaTile[0][threadIdx.x +  2];
		__syncthreads();
	}
	if(threadIdx.x < 1) {
		ThetaTile[0][threadIdx.x] += ThetaTile[0][threadIdx.x +  1];
		__syncthreads();
	}
	
	atomicAdd(result, ThetaTile[0][0]);
}

__global__ void SigmoidGradient(Matrix Z)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float g = 1 / (1 + exp(-Z.elements[Row * Z.pitch + Col]));
	Z.elements[Row * Z.pitch + Col] = g * (1 - g);
}

__global__ void GradProcessing(Matrix Theta_grad, Matrix Delta, Matrix Theta, float lambda, int m)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int width =  Delta.width - 1;
	
	if(Col < width) {
		Theta_grad.elements[Row * Theta_grad.pitch + Col] = (Delta.elements[Row * Delta.pitch + Col] 
															+ lambda * Theta.elements[Row * Theta.pitch + Col]) / m;
	}
	
	Theta_grad.elements[Row * Theta_grad.pitch ] = Delta.elements[Row * Delta.pitch] / m;
}
#endif // #ifndef _MATRIXMUL_KERNEL_H_
