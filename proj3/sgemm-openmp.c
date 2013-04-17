#include <stdio.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <omp.h>

// cache line size 64 bytes / float size 4 bytes = block size (BS) 16 floats
#define BS 8


void sgemm( int m, int n, int d, float *A, float *C )
{

	/*
	//Original Code

	for( int i = 0; i < n; i++ )
		for( int k = 0; k < m; k++ ) 
    		for( int j = 0; j < n; j++ ) 
				C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
	*/

	#pragma omp parallel
	{
		int num_t = omp_get_num_threads();
		int t_id = omp_get_thread_num();
		int M = (m-(m%BS));
		int t_block = n/num_t + 1;
		int startJ = t_id * t_block;
		int endJ = startJ + t_block;
		if (endJ <= n) {
			for( int j = startJ; j < endJ; j++ ) {
				for( int k = 0; k < (m-m%4); k += 4 ) { 
					float* basej = A + j * (n + 1);
					__m128 transposeValue0 = _mm_load1_ps(basej + k * n);
					__m128 transposeValue1 = _mm_load1_ps(basej + (k+1) * n);
					__m128 transposeValue2 = _mm_load1_ps(basej + (k+2) * n);
					__m128 transposeValue3 = _mm_load1_ps(basej + (k+3) * n);
					for( int i = 0; i < (n - n%16); i += 16 ) {
						float* position = C + i + j * n;
						__m128 storedValues1 = _mm_loadu_ps(position);
						__m128 storedValues2 = _mm_loadu_ps(position + 4);
						__m128 storedValues3 = _mm_loadu_ps(position + 8);
						__m128 storedValues4 = _mm_loadu_ps(position + 12);
						float* basei0 = A + i;
						float* basei4 = basei0 + 4;
						float* basei8 = basei4 + 4;
						float* basei12 = basei8 + 4;
						__m128 columnValues1;
						__m128 columnValues2;
						__m128 columnValues3;
						__m128 columnValues4;

						//computation 1
						columnValues1 = _mm_loadu_ps(basei0 + k * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + k * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + k * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + k * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

						//computation 2
						columnValues1 = _mm_loadu_ps(basei0 + (k+1) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue1, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+1) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue1, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+1) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue1, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+1) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue1, columnValues4));

						//computation 3
						columnValues1 = _mm_loadu_ps(basei0 + (k+2) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue2, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+2) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue2, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+2) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue2, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+2) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue2, columnValues4));

						//computation 4
						columnValues1 = _mm_loadu_ps(basei0 + (k+3) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue3, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+3) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue3, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+3) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue3, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+3) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue3, columnValues4));

						
						_mm_storeu_ps(position, storedValues1);
						_mm_storeu_ps(position + 4, storedValues2);
						_mm_storeu_ps(position + 8, storedValues3);
						_mm_storeu_ps(position + 12, storedValues4);


						//C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
					for (int i = (n - n%16); i < n; i++) {
						C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
						C[i+j*n] += A[i+(k+1)*(n)] * A[j*(n+1)+(k+1)*(n)];
						C[i+j*n] += A[i+(k+2)*(n)] * A[j*(n+1)+(k+2)*(n)];
						C[i+j*n] += A[i+(k+3)*(n)] * A[j*(n+1)+(k+3)*(n)];
					}
				}
				for( int k = (m-m%4); k < m; k += 1 ) { 
					float* basej = A + j * (n + 1);
					__m128 transposeValue0 = _mm_load1_ps(basej + k * n);
					for( int i = 0; i < (n - n%16); i += 16 ) {
						float* position = C + i + j * n;
						__m128 storedValues1 = _mm_loadu_ps(position);
						__m128 storedValues2 = _mm_loadu_ps(position + 4);
						__m128 storedValues3 = _mm_loadu_ps(position + 8);
						__m128 storedValues4 = _mm_loadu_ps(position + 12);
						float* basei0 = A + i;
						float* basei4 = basei0 + 4;
						float* basei8 = basei4 + 4;
						float* basei12 = basei8 + 4;
						__m128 columnValues1;
						__m128 columnValues2;
						__m128 columnValues3;
						__m128 columnValues4;


						columnValues1 = _mm_loadu_ps(basei0 + k * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + k * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + k * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + k * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

						
						_mm_storeu_ps(position, storedValues1);
						_mm_storeu_ps(position + 4, storedValues2);
						_mm_storeu_ps(position + 8, storedValues3);
						_mm_storeu_ps(position + 12, storedValues4);


						//C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
					for (int i = (n - n%16); i < n; i++) {
						C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
				}
			}
		} else if (startJ < n) {
			for( int j = startJ; j < n; j++ ) {
				for( int k = 0; k < (m-m%4); k += 4 ) { 
					float* basej = A + j * (n + 1);
					__m128 transposeValue0 = _mm_load1_ps(basej + k * n);
					__m128 transposeValue1 = _mm_load1_ps(basej + (k+1) * n);
					__m128 transposeValue2 = _mm_load1_ps(basej + (k+2) * n);
					__m128 transposeValue3 = _mm_load1_ps(basej + (k+3) * n);
					for( int i = 0; i < (n - n%16); i += 16 ) {
						float* position = C + i + j * n;
						__m128 storedValues1 = _mm_loadu_ps(position);
						__m128 storedValues2 = _mm_loadu_ps(position + 4);
						__m128 storedValues3 = _mm_loadu_ps(position + 8);
						__m128 storedValues4 = _mm_loadu_ps(position + 12);
						float* basei0 = A + i;
						float* basei4 = basei0 + 4;
						float* basei8 = basei4 + 4;
						float* basei12 = basei8 + 4;
						__m128 columnValues1;
						__m128 columnValues2;
						__m128 columnValues3;
						__m128 columnValues4;

						//computation 1
						columnValues1 = _mm_loadu_ps(basei0 + k * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + k * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + k * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + k * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

						//computation 2
						columnValues1 = _mm_loadu_ps(basei0 + (k+1) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue1, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+1) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue1, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+1) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue1, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+1) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue1, columnValues4));

						//computation 3
						columnValues1 = _mm_loadu_ps(basei0 + (k+2) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue2, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+2) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue2, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+2) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue2, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+2) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue2, columnValues4));

						//computation 4
						columnValues1 = _mm_loadu_ps(basei0 + (k+3) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue3, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (k+3) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue3, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (k+3) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue3, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (k+3) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue3, columnValues4));

						
						_mm_storeu_ps(position, storedValues1);
						_mm_storeu_ps(position + 4, storedValues2);
						_mm_storeu_ps(position + 8, storedValues3);
						_mm_storeu_ps(position + 12, storedValues4);


						//C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
					for (int i = (n - n%16); i < n; i++) {
						C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
						C[i+j*n] += A[i+(k+1)*(n)] * A[j*(n+1)+(k+1)*(n)];
						C[i+j*n] += A[i+(k+2)*(n)] * A[j*(n+1)+(k+2)*(n)];
						C[i+j*n] += A[i+(k+3)*(n)] * A[j*(n+1)+(k+3)*(n)];
					}
				}
				for( int k = (m-m%4); k < m; k += 1 ) { 
					float* basej = A + j * (n + 1);
					__m128 transposeValue0 = _mm_load1_ps(basej + k * n);
					for( int i = 0; i < (n - n%16); i += 16 ) {
						float* position = C + i + j * n;
						__m128 storedValues1 = _mm_loadu_ps(position);
						__m128 storedValues2 = _mm_loadu_ps(position + 4);
						__m128 storedValues3 = _mm_loadu_ps(position + 8);
						__m128 storedValues4 = _mm_loadu_ps(position + 12);
						float* basei0 = A + i;
						float* basei4 = basei0 + 4;
						float* basei8 = basei4 + 4;
						float* basei12 = basei8 + 4;
						__m128 columnValues1;
						__m128 columnValues2;
						__m128 columnValues3;
						__m128 columnValues4;


						columnValues1 = _mm_loadu_ps(basei0 + k * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + k * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + k * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + k * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

						
						_mm_storeu_ps(position, storedValues1);
						_mm_storeu_ps(position + 4, storedValues2);
						_mm_storeu_ps(position + 8, storedValues3);
						_mm_storeu_ps(position + 12, storedValues4);


						//C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
					for (int i = (n - n%16); i < n; i++) {
						C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
					}
				}
			}
		}
	}
}
