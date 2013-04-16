#include <stdio.h>
#include <emmintrin.h>
#include <xmmintrin.h>

// cache line size 64 bytes / float size 4 bytes = block size (BS) 16 floats
#define BS 16


void sgemm( int m, int n, int d, float *A, float *C )
{

	/*
	//Original Code

	for( int i = 0; i < n; i++ )
		for( int k = 0; k < m; k++ ) 
    		for( int j = 0; j < n; j++ ) 
				C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
	*/

	/*
	//First Blocking Implementation

	int N = (n-(n%BS)), M = (m-(m%BS));
	for (int K = 0; K < M; K += BS) {
		for (int J = 0; J < N; J += BS) {
			for (int I = 0; I < N; I += BS) {
				// Blocks
				for (int j = 0; j < BS; j += 1) {
					for (int k = 0; k < BS; k += 1) {
						for (int i = 0; i < BS; i += 1) {
							C[(I+i)+(J+j)*n] += A[(I+i)+(K+k)*n] * A[(J+j)*(n+1)+(K+k)*n];
						}
					}
				}
			}
		}
	}

	*/		
	int N = (n-(n%BS)), M = (m-(m%BS));
	for (int K = 0; K < M; K += BS) {
		for (int J = 0; J < N; J += BS) {
			for (int I = 0; I < N; I += BS) {
				// Blocks
				float* basei0 = A + I;
				float* basei4 = basei0 + 4;
				float* basei8 = basei4 + 4;
				float* basei12 = basei8 + 4;
				for (int j = 0; j < BS; j += 1) {
					float* basej = A + (J+j)*(n+1);
					float* position = C+(I)+(J+j)*n;

					//-----------
					for (int k = 0; k < BS; k += 8) {
						__m128 transposeValue0 = _mm_load1_ps(basej + (K+k) * n);
						__m128 transposeValue1 = _mm_load1_ps(basej + (K+k+1) * n);
						__m128 transposeValue2 = _mm_load1_ps(basej + (K+k+2) * n);
						__m128 transposeValue3 = _mm_load1_ps(basej + (K+k+3) * n);
						__m128 transposeValue4 = _mm_load1_ps(basej + (K+k+4) * n);
						__m128 transposeValue5 = _mm_load1_ps(basej + (K+k+5) * n);
						__m128 transposeValue6 = _mm_load1_ps(basej + (K+k+6) * n);
						__m128 transposeValue7 = _mm_load1_ps(basej + (K+k+7) * n);

						
						__m128 storedValues1 = _mm_loadu_ps(position);
						__m128 storedValues2 = _mm_loadu_ps(position + 4);
						__m128 storedValues3 = _mm_loadu_ps(position + 8);
						__m128 storedValues4 = _mm_loadu_ps(position + 12);

						__m128 columnValues1;
						__m128 columnValues2;
						__m128 columnValues3;
						__m128 columnValues4;

						//comoutation 1
						columnValues1 = _mm_loadu_ps(basei0 + (K+k) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

						//comoutation 2
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+1) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue1, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+1) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue1, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+1) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue1, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+1) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue1, columnValues4));

						//comoutation 3
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+2) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue2, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+2) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue2, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+2) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue2, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+2) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue2, columnValues4));

						//comoutation 4
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+3) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue3, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+3) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue3, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+3) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue3, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+3) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue3, columnValues4));

						//comoutation 5
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+4) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue4, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+4) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue4, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+4) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue4, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+4) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue4, columnValues4));

						//comoutation 6
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+5) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue5, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+5) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue5, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+5) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue5, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+5) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue5, columnValues4));

						//comoutation 7
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+6) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue6, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+6) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue6, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+6) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue6, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+6) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue6, columnValues4));

						//comoutation 8
						columnValues1 = _mm_loadu_ps(basei0 + (K+k+7) * n);
						storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue7, columnValues1));

						columnValues2 = _mm_loadu_ps(basei4 + (K+k+7) * n);
						storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue7, columnValues2));

						columnValues3 = _mm_loadu_ps(basei8 + (K+k+7) * n);
						storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue7, columnValues3));

						columnValues4 = _mm_loadu_ps(basei12 + (K+k+7) * n);
						storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue7, columnValues4));



						_mm_storeu_ps(position, storedValues1);
						_mm_storeu_ps(position + 4, storedValues2);
						_mm_storeu_ps(position + 8, storedValues3);
						_mm_storeu_ps(position + 12, storedValues4);
					}
					//-----------
				}
			}
		}
	}
}

					//Further unrolling of k. performance got worse.
					//------
					/*
					__m128 transposeValue0 = _mm_load1_ps(basej + (K) * n);
					__m128 transposeValue1 = _mm_load1_ps(basej + (K+1) * n);
					__m128 transposeValue2 = _mm_load1_ps(basej + (K+2) * n);
					__m128 transposeValue3 = _mm_load1_ps(basej + (K+3) * n);
					__m128 transposeValue4 = _mm_load1_ps(basej + (K+4) * n);
					__m128 transposeValue5 = _mm_load1_ps(basej + (K+5) * n);
					__m128 transposeValue6 = _mm_load1_ps(basej + (K+6) * n);
					__m128 transposeValue7 = _mm_load1_ps(basej + (K+7) * n);
					__m128 transposeValue8 = _mm_load1_ps(basej + (K+8) * n);
					__m128 transposeValue9 = _mm_load1_ps(basej + (K+9) * n);
					__m128 transposeValue10 = _mm_load1_ps(basej + (K+10) * n);
					__m128 transposeValue11 = _mm_load1_ps(basej + (K+11) * n);
					__m128 transposeValue12 = _mm_load1_ps(basej + (K+12) * n);
					__m128 transposeValue13 = _mm_load1_ps(basej + (K+13) * n);
					__m128 transposeValue14 = _mm_load1_ps(basej + (K+14) * n);
					__m128 transposeValue15 = _mm_load1_ps(basej + (K+15) * n);

					
					__m128 storedValues1 = _mm_loadu_ps(position);
					__m128 storedValues2 = _mm_loadu_ps(position + 4);
					__m128 storedValues3 = _mm_loadu_ps(position + 8);
					__m128 storedValues4 = _mm_loadu_ps(position + 12);

					__m128 columnValues1;
					__m128 columnValues2;
					__m128 columnValues3;
					__m128 columnValues4;

					//comoutation 1
					columnValues1 = _mm_loadu_ps(basei0 + (K) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue0, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue0, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue0, columnValues4));

					//comoutation 2
					columnValues1 = _mm_loadu_ps(basei0 + (K+1) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue1, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+1) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue1, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+1) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue1, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+1) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue1, columnValues4));

					//comoutation 3
					columnValues1 = _mm_loadu_ps(basei0 + (K+2) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue2, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+2) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue2, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+2) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue2, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+2) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue2, columnValues4));

					//comoutation 4
					columnValues1 = _mm_loadu_ps(basei0 + (K+3) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue3, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+3) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue3, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+3) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue3, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+3) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue3, columnValues4));

					//comoutation 5
					columnValues1 = _mm_loadu_ps(basei0 + (K+4) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue4, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+4) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue4, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+4) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue4, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+4) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue4, columnValues4));

					//comoutation 6
					columnValues1 = _mm_loadu_ps(basei0 + (K+5) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue5, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+5) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue5, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+5) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue5, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+5) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue5, columnValues4));

					//comoutation 7
					columnValues1 = _mm_loadu_ps(basei0 + (K+6) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue6, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+6) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue6, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+6) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue6, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+6) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue6, columnValues4));

					//comoutation 8
					columnValues1 = _mm_loadu_ps(basei0 + (K+7) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue7, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+7) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue7, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+7) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue7, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+7) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue7, columnValues4));

					//comoutation 9
					columnValues1 = _mm_loadu_ps(basei0 + (K+8) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue8, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+8) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue8, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+8) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue8, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+8) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue8, columnValues4));

					//comoutation 10
					columnValues1 = _mm_loadu_ps(basei0 + (K+9) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue9, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+9) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue9, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+9) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue9, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+9) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue9, columnValues4));

					//comoutation 11
					columnValues1 = _mm_loadu_ps(basei0 + (K+10) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue10, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+10) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue10, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+10) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue10, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+10) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue10, columnValues4));

					//comoutation 12
					columnValues1 = _mm_loadu_ps(basei0 + (K+11) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue11, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+11) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue11, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+11) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue11, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+11) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue11, columnValues4));

					//comoutation 13
					columnValues1 = _mm_loadu_ps(basei0 + (K+12) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue12, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+12) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue12, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+12) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue12, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+12) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue12, columnValues4));

					//comoutation 14
					columnValues1 = _mm_loadu_ps(basei0 + (K+13) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue13, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+13) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue13, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+13) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue13, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+13) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue13, columnValues4));

					//comoutation 15
					columnValues1 = _mm_loadu_ps(basei0 + (K+14) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue14, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+14) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue14, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+14) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue14, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+14) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue14, columnValues4));

					//comoutation 16
					columnValues1 = _mm_loadu_ps(basei0 + (K+15) * n);
					storedValues1 = _mm_add_ps(storedValues1, _mm_mul_ps(transposeValue15, columnValues1));

					columnValues2 = _mm_loadu_ps(basei4 + (K+15) * n);
					storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue15, columnValues2));

					columnValues3 = _mm_loadu_ps(basei8 + (K+15) * n);
					storedValues3 = _mm_add_ps(storedValues3, _mm_mul_ps(transposeValue15, columnValues3));

					columnValues4 = _mm_loadu_ps(basei12 + (K+15) * n);
					storedValues4 = _mm_add_ps(storedValues4, _mm_mul_ps(transposeValue15, columnValues4));



					_mm_storeu_ps(position, storedValues1);
					_mm_storeu_ps(position + 4, storedValues2);
					_mm_storeu_ps(position + 8, storedValues3);
					_mm_storeu_ps(position + 12, storedValues4);

					*/
					//-------



