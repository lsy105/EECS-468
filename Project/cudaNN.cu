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
 * Host code.
 */

//define tile size
#define TILE_SIZE 32
#define HIDDEN_LAYER_SIZE 25
#define NUM_LABLES 10

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <cudaNN_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void BPNetWork(Matrix &X, Matrix &y);
void Trainning(Matrix &Theta1, Matrix &Theta2, Matrix &X, Matrix &y);
void predict(Matrix &Theta1, Matrix &Theta2, Matrix &X, Matrix &y);
void nnCostFunction(Matrix &Theta1d, Matrix &Theta2d, Matrix &X, Matrix &y, int num_labels, float lambda, Matrix &Theta1Gradd, Matrix &Theta2Gradd);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	
	unsigned int timer;
	int errorM = 0, errorN = 0;
	
	srand(52);
	
	//Random initialization
	Matrix X  = AllocateMatrix(5000, 400, 0);
	Matrix y  = AllocateMatrix(5000, 1, 0);
	errorM = ReadFile(&X, "X.txt");
	errorN = ReadFile(&y, "y.txt");
	
	if(errorM  || errorN )
	{
		printf("Error reading input files %d, %d\n", errorM, errorN);
		return 1;
	}
	
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutStartTimer(timer);
    BPNetWork(X, y);
	cutStopTimer(timer);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Device GPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
	
	// Free matrices
	FreeMatrix(&X);
	FreeMatrix(&y);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a BPNetWork test on CUDA
////////////////////////////////////////////////////////////////////////////////
void BPNetWork(Matrix &X, Matrix &y)
{
	//Trainning
	Matrix Theta1 = AllocateMatrix(HIDDEN_LAYER_SIZE, X.width + 1, 1);
	Matrix Theta2 = AllocateMatrix(NUM_LABLES, HIDDEN_LAYER_SIZE + 1, 1);
	Trainning(Theta1, Theta2, X, y);
	
	//Predict
	//predict(Theta1, Theta2, X, y);
}

void Trainning(Matrix &Theta1, Matrix &Theta2, Matrix &X, Matrix &y)
{
	float lambda = 1;
	int Grid_x = 0;
	int Grid_y = 0;
	dim3 dimGrid(0, 0);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	
	//RandomInitialization
	Matrix Theta1d = AllocateDeviceMatrix(Theta1);
	CopyToDeviceMatrix(Theta1d, Theta1);
	
	Matrix Theta2d = AllocateDeviceMatrix(Theta2);
	CopyToDeviceMatrix(Theta2d, Theta2);
	
	Matrix Theta1Grad = AllocateMatrix(HIDDEN_LAYER_SIZE, X.width + 1, 1);
	Matrix Theta1Gradd = AllocateDeviceMatrix(Theta1Grad);
	CopyToDeviceMatrix(Theta1Gradd, Theta1Grad);
	
	Matrix Theta2Grad = AllocateMatrix(NUM_LABLES, HIDDEN_LAYER_SIZE + 1, 1);
	Matrix Theta2Gradd = AllocateDeviceMatrix(Theta2Grad);
	CopyToDeviceMatrix(Theta2Gradd, Theta2Grad);
	
	for(int i = 0; i < 50; i++) {
		//costFunction
		nnCostFunction(Theta1d, Theta2d, X, y, NUM_LABLES, lambda, Theta1Gradd, Theta2Gradd);
		
		//Gradient Descent
		Grid_x = Theta1.width/TILE_SIZE + ((Theta1.width%TILE_SIZE) ? 1 : 0);
		Grid_y = Theta1.height/TILE_SIZE + ((Theta1.height%TILE_SIZE) ? 1 : 0);
	
		dimGrid.x = Grid_x;
		dimGrid.y = Grid_y;
		MatrixSubKernel<<<dimGrid, dimBlock>>>(Theta1d, Theta1Gradd, Theta1d);
		
		Grid_x = Theta2.width/TILE_SIZE + ((Theta2.width%TILE_SIZE) ? 1 : 0);
		Grid_y = Theta2.height/TILE_SIZE + ((Theta2.height%TILE_SIZE) ? 1 : 0);
	
		dimGrid.x = Grid_x;
		dimGrid.y = Grid_y;
		MatrixSubKernel<<<dimGrid, dimBlock>>>(Theta2d, Theta2Gradd, Theta2d);
	}
	
	CopyFromDeviceMatrix(Theta1, Theta1d);
	CopyFromDeviceMatrix(Theta2, Theta2d);
	
	FreeDeviceMatrix(&Theta1d);
	FreeDeviceMatrix(&Theta2d);
	FreeDeviceMatrix(&Theta1Gradd);
	FreeDeviceMatrix(&Theta2Gradd);
}

void predict(Matrix &Theta1, Matrix &Theta2, Matrix &X, Matrix &y)
{
	int Grid_x = 0;
	int Grid_y = 0;
	dim3 dimGrid(0, 0);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	
	Matrix Theta1d = AllocateDeviceMatrix(Theta1);
	CopyToDeviceMatrix(Theta1d, Theta1);
	
	Matrix Theta2d = AllocateDeviceMatrix(Theta2);
	CopyToDeviceMatrix(Theta2d, Theta2);
	
	Matrix a1 = AllocateDeviceMatrix(X);
	CopyToDeviceMatrix(a1, X);
	
	Matrix HiddenLayer = AllocateMatrix(X.height, Theta1.height, 0);
	Matrix a2 = AllocateDeviceMatrix(HiddenLayer);
	CopyToDeviceMatrix(a2, HiddenLayer);
	
	Matrix OutputLayer = AllocateMatrix(HiddenLayer.height, Theta2.width, 0);
	Matrix a3 = AllocateDeviceMatrix(OutputLayer);
	CopyToDeviceMatrix(a3, OutputLayer);
	
	/********** ForwardPropagateBetweenLayers **********/
	
	// Setup the execution configuration
	Grid_x = HiddenLayer.width/TILE_SIZE + ((HiddenLayer.width%TILE_SIZE) ? 1 : 0);
	Grid_y = HiddenLayer.height/TILE_SIZE + ((HiddenLayer.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	ForwardPropagateBetweenLayers<<<dimGrid, dimBlock>>>(a1, a2, Theta1d);
	
	// Setup the execution configuration
	Grid_x = OutputLayer.width/TILE_SIZE + ((OutputLayer.width%TILE_SIZE) ? 1 : 0);
	Grid_y = OutputLayer.height/TILE_SIZE + ((OutputLayer.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;;
	
	// Launch the device computation threads!
	ForwardPropagateBetweenLayers<<<dimGrid, dimBlock>>>(a2, a3, Theta2d);
}

void nnCostFunction(Matrix &Theta1d, Matrix &Theta2d, Matrix &X, Matrix &y, int num_labels, float lambda, Matrix &Theta1Gradd, Matrix &Theta2Gradd)
{
	//Initialize variable
	int Grid_x = 0;
	int Grid_y = 0;
	dim3 dimGrid(0, 0);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	float *Jresult = (float*) malloc(sizeof(float));
	float *Theta1result = (float*) malloc(sizeof(float));
	float *Theta2result = (float*) malloc(sizeof(float));
	float *Jresultd = (float*) malloc(sizeof(float));
	float *Theta1resultd = (float*) malloc(sizeof(float));
	float *Theta2resultd = (float*) malloc(sizeof(float));
	*Jresult = *Theta1result = *Theta2result = 0;
	cudaMalloc((void**)&Jresultd, sizeof(float));
	cudaMalloc((void**)&Theta1resultd, sizeof(float));
	cudaMalloc((void**)&Theta2resultd, sizeof(float));
	cudaMemcpy(&Jresultd, &Jresult, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&Theta1resultd, &Theta1result, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&Theta2resultd, &Theta2result, sizeof(float), cudaMemcpyHostToDevice);
	
	/********** Initialize and allocate Matrix on device **********/
	Matrix a1 = AllocateDeviceMatrix(X);
	CopyToDeviceMatrix(a1, X);
	
	Matrix HiddenLayer = AllocateMatrix(X.height, Theta1d.height, 0);
	Matrix a2 = AllocateDeviceMatrix(HiddenLayer);
	CopyToDeviceMatrix(a2, HiddenLayer);
	
	Matrix OutputLayer = AllocateMatrix(HiddenLayer.height, Theta2d.width, 0);
	Matrix a3 = AllocateDeviceMatrix(OutputLayer);
	CopyToDeviceMatrix(a3, OutputLayer);
	
	Matrix yd = AllocateDeviceMatrix(y);
	CopyToDeviceMatrix(yd, y);
	
	Matrix y_vector = AllocateMatrix(X.height, num_labels, 0);//TODO:height or width
	Matrix y_vectord = AllocateDeviceMatrix(y_vector);
	CopyToDeviceMatrix(y_vectord, y_vector);
	
	Matrix J = AllocateMatrix(OutputLayer.height, OutputLayer.width, 0);
	Matrix Jd = AllocateDeviceMatrix(J);
	CopyToDeviceMatrix(Jd, J);
	
	Matrix z2 = AllocateMatrix(X.height, Theta1d.height, 0);
	Matrix z2d = AllocateDeviceMatrix(z2);
	CopyToDeviceMatrix(z2d, z2);
	
	Matrix delta2 = AllocateMatrix(HiddenLayer.height, HiddenLayer.width, 0);
	Matrix delta2d = AllocateDeviceMatrix(delta2);
	CopyToDeviceMatrix(delta2d, delta2);
	
	Matrix delta3 = AllocateMatrix(OutputLayer.height, OutputLayer.width, 0);
	Matrix delta3d = AllocateDeviceMatrix(delta3);
	CopyToDeviceMatrix(delta3d, delta3);
	
	Matrix P = AllocateMatrix(delta3.height, Theta2d.width, 0);// P = delta3 * Theta2
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P);
	
	Matrix Delta1 = AllocateMatrix(Theta1d.height, Theta1d.width, 0);
	Matrix Delta1d = AllocateDeviceMatrix(Delta1);
	CopyToDeviceMatrix(Delta1d, Delta1);
	
	Matrix Delta2 = AllocateMatrix(Theta2d.height, Theta2d.width, 0);
	Matrix Delta2d = AllocateDeviceMatrix(Delta2);
	CopyToDeviceMatrix(Delta2d, Delta2);
	
	/********** ForwardPropagateBetweenLayers **********/
	// Setup the execution configuration
	Grid_x = HiddenLayer.width/TILE_SIZE + ((HiddenLayer.width%TILE_SIZE) ? 1 : 0);
	Grid_y = HiddenLayer.height/TILE_SIZE + ((HiddenLayer.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	ForwardPropagateBetweenLayers<<<dimGrid, dimBlock>>>(a1, a2, Theta1d);
	
	// Setup the execution configuration
	Grid_x = OutputLayer.width/TILE_SIZE + ((OutputLayer.width%TILE_SIZE) ? 1 : 0);
	Grid_y = OutputLayer.height/TILE_SIZE + ((OutputLayer.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;;
	
	// Launch the device computation threads!
	ForwardPropagateBetweenLayers<<<dimGrid, dimBlock>>>(a2, a3, Theta2d);

	/********** BuildSampleVector **********/
	// Setup the execution configuration
	Grid_x = y_vector.width/TILE_SIZE + ((y_vector.width%TILE_SIZE) ? 1 : 0);
	Grid_y = y_vector.height/TILE_SIZE + ((y_vector.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	buildVector<<<dimGrid, dimBlock>>>(yd, y_vectord);
	
	/********** ComputeJ **********/
	// Setup the execution configuration
	Grid_x = J.width/TILE_SIZE + ((J.width%TILE_SIZE) ? 1 : 0);
	Grid_y = J.height/TILE_SIZE + ((J.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	ComputeJ<<<dimGrid, dimBlock>>>(Jd, y_vectord, a3, Theta1d, Theta2d, lambda);
	
	// Launch the device computation threads!
	TwoDimSum<<<dimGrid, dimBlock>>>(Jd, Jresultd);
	
	// Setup the execution configuration
	Grid_x = Theta1d.width/TILE_SIZE + ((Theta1d.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Theta1d.height/TILE_SIZE + ((Theta1d.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	TwoDimSumTheta<<<dimGrid, dimBlock>>>(Theta1d, Theta1resultd);
	
	// Setup the execution configuration
	Grid_x = Theta2d.width/TILE_SIZE + ((Theta2d.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Theta2d.height/TILE_SIZE + ((Theta2d.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	TwoDimSumTheta<<<dimGrid, dimBlock>>>(Theta2d, Theta2resultd);
	
	//Copy result from device to host
	cudaMemcpy(&Jresult, &Jresultd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Theta1result, &Theta1resultd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Theta2result, &Theta2resultd, sizeof(float), cudaMemcpyDeviceToHost);
	
	//Add result
	*Jresult += lambda*(*Theta1result + *Theta2result)/2*X.height;
	
	/********** Compute ThetaGradient **********/
	// Setup the execution configuration
	Grid_x = z2.width/TILE_SIZE + ((z2.width%TILE_SIZE) ? 1 : 0);
	Grid_y = z2.height/TILE_SIZE + ((z2.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	MatrixMulTransposePaddingKernel<<<dimGrid, dimBlock>>>(a1, Theta1d, z2d);
	
	//Compute Delta
	Grid_x = delta3d.width/TILE_SIZE + ((delta3d.width%TILE_SIZE) ? 1 : 0);
	Grid_y = delta3d.height/TILE_SIZE + ((delta3d.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	MatrixSubKernel<<<dimGrid, dimBlock>>>(a3, y_vector, delta3d);
	
	// Setup the execution configuration
	Grid_x = z2.width/TILE_SIZE + ((z2.width%TILE_SIZE) ? 1 : 0);
	Grid_y = z2.height/TILE_SIZE + ((z2.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	SigmoidGradient<<<dimGrid, dimBlock>>>(z2d);
	
	// Setup the execution configuration
	Grid_x = Pd.width/TILE_SIZE + ((Pd.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Pd.height/TILE_SIZE + ((Pd.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	MatrixMulKernel<<<dimGrid, dimBlock>>>(delta3d, Theta2d, Pd);
	
	// Setup the execution configuration
	Grid_x = delta2.width/TILE_SIZE + ((delta2.width%TILE_SIZE) ? 1 : 0);
	Grid_y = delta2.height/TILE_SIZE + ((delta2.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	SubMatrixDotMulKernel<<<dimGrid, dimBlock>>>(Pd, z2d, delta2d);
	
	// Setup the execution configuration
	Grid_x = Delta2.width/TILE_SIZE + ((Delta2.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Delta2.height/TILE_SIZE + ((Delta2.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	MatrixTransposeMulKernel<<<dimGrid, dimBlock>>>(delta3d, a2, Delta2);
	
	// Setup the execution configuration
	Grid_x = Delta1.width/TILE_SIZE + ((Delta1.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Delta1.height/TILE_SIZE + ((Delta1.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	MatrixTransposeMulKernel<<<dimGrid, dimBlock>>>(delta2d, a1, Delta1d);
	
	// Setup the execution configuration
	Grid_x = Theta2d.width/TILE_SIZE + ((Theta2d.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Theta2d.height/TILE_SIZE + ((Theta2d.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	GradProcessing<<<dimGrid, dimBlock>>>(Theta2Gradd, Delta2, Theta2d, lambda, X.height);
	
	// Setup the execution configuration
	Grid_x = Theta1Gradd.width/TILE_SIZE + ((Theta1Gradd.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Theta1Gradd.height/TILE_SIZE + ((Theta1Gradd.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	GradProcessing<<<dimGrid, dimBlock>>>(Theta1Gradd, Delta1d, Theta1d, lambda, X.height);
	
	// Setup the execution configuration
	Grid_x = Theta2Gradd.width/TILE_SIZE + ((Theta2Gradd.width%TILE_SIZE) ? 1 : 0);
	Grid_y = Theta2Gradd.height/TILE_SIZE + ((Theta2Gradd.height%TILE_SIZE) ? 1 : 0);
	
	dimGrid.x = Grid_x;
	dimGrid.y = Grid_y;
	
	// Launch the device computation threads!
	GradProcessing<<<dimGrid, dimBlock>>>(Theta2Gradd, Delta2d, Theta2d, lambda, X.height);
	
	/********** Free Device Matrices **********/
	FreeDeviceMatrix(&a1);
	FreeDeviceMatrix(&a2);
	FreeDeviceMatrix(&a3);
	FreeDeviceMatrix(&yd);
	FreeDeviceMatrix(&y_vectord);
	FreeDeviceMatrix(&Jd);
	FreeDeviceMatrix(&z2d);
	FreeDeviceMatrix(&delta2d);
	FreeDeviceMatrix(&delta3d);
	FreeDeviceMatrix(&Delta1d);
	FreeDeviceMatrix(&Delta2d);
	FreeDeviceMatrix(&Pd);
}

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	int Grid_x, Grid_y;
	
	// Load M and N to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory
	
	// Setup the execution configuration
	Grid_x = P.width/TILE_SIZE + ((P.width%TILE_SIZE) ? 1 : 0);
	Grid_y = P.height/TILE_SIZE + ((P.height%TILE_SIZE) ? 1 : 0);
	
	dim3 dimGrid(Grid_x, Grid_y);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	
	// Launch the device computation threads!
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
	
	// Read P from the device
	CopyFromDeviceMatrix(P, Pd);

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

void MatrixDotMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	int Grid_x, Grid_y;
	
	// Load M and N to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory
	
	// Setup the execution configuration
	Grid_x = P.width/TILE_SIZE + ((P.width%TILE_SIZE) ? 1 : 0);
	Grid_y = P.height/TILE_SIZE + ((P.height%TILE_SIZE) ? 1 : 0);
	
	dim3 dimGrid(Grid_x, Grid_y);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	
	// Launch the device computation threads!
	MatrixDotMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
	
	// Read P from the device
	CopyFromDeviceMatrix(P, Pd);

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

void MatrixConstMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	int Grid_x, Grid_y;
	
	// Load M and N to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory
	
	// Setup the execution configuration
	Grid_x = P.width/TILE_SIZE + ((P.width%TILE_SIZE) ? 1 : 0);
	Grid_y = P.height/TILE_SIZE + ((P.height%TILE_SIZE) ? 1 : 0);
	
	dim3 dimGrid(Grid_x, Grid_y);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	
	// Launch the device computation threads!
	MatrixDotMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
	
	// Read P from the device
	CopyFromDeviceMatrix(P, Pd);

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height*M->width;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return (data_read != (M->height * M->width));
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    cutWriteFilef(file_name, M.elements, M.width*M.height,
                       0.0001f);
}
