#include <stdio.h>
#include <emmintrin.h>
#include <xmmintrin.h>

/* Size of blocks. Cache line / sizeof float. 64 bytes / 4 bytes = 16 floats. */
int BLOCK_WIDTH = 16;




/* Naive operation. M is the size of the strip. N is the number of rows of A. 
   N+D is the number of colums of A. */
void sgemm( int m, int n, int d, float *A, float *C )
{
 
for( int k = 0; k < m; k++ ) {
	for( int j = 0; j < n; j++ ) {
      	for( int i = 0; i < (n - (n % 4)); i += 4 ) {
			// C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
			// Unrolling i-loop
			__m128 value1 = _mm_loadu_ps (A + i+k*n);
			__m128 value2 = _mm_load1_ps (A + j*(n+1)+k*(n));
			__m128 intValue = _mm_mul_ps (value1, value2);
			__m128 oldValue = _mm_loadu_ps (C + i+j*n);
			__m128 newValue = _mm_add_ps (intValue, Value);
			_mm_storeu_ps(C + i+j*n, newValue);
		}
		for ( int i = (n - (n % 4)); i < n; i++)
			C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
		}
	} 
}

/* step1: loop unrolling with ssid instructions. */

/* Padding or edge cases? */