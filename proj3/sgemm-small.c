#include <stdio.h>
#include <emmintrin.h>
#include <xmmintrin.h>


void sgemm( int m, int n, int d, float *A, float *C )
{
	for (int j = 0; j < n; j++) {
		for (int k = 0; k < m; k += 4) {
			float* basej = A + j * (n + 1);
			__m128 transposeValue0 = _mm_load1_ps(basej + k * n);
			__m128 transposeValue1 = _mm_load1_ps(basej + (k + 1) * n);
			__m128 transposeValue2 = _mm_load1_ps(basej + (k + 2) * n);
			__m128 transposeValue3 = _mm_load1_ps(basej + (k + 3) * n);

			__m128 storedValues;
			__m128 columnValues;
			__m128 storedValues2;
			__m128 columnValues2;
			for (int i = 0; i < n; i += 8) {
				float* position = C + i + j * n;
				storedValues = _mm_loadu_ps(position);
				storedValues2 = _mm_loadu_ps(position + 4);

				//first computation
				columnValues = _mm_loadu_ps(A + i + k * n);
				storedValues = _mm_add_ps(storedValues, _mm_mul_ps(transposeValue0, columnValues));

				columnValues2 = _mm_loadu_ps(A + (i + 4) + k * n);
				storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue0, columnValues2));

				//second computation
				columnValues = _mm_loadu_ps(A + i + (k + 1) * n);
				storedValues = _mm_add_ps(storedValues, _mm_mul_ps(transposeValue1, columnValues));

				columnValues2 = _mm_loadu_ps(A + (i + 4) + (k + 1) * n);
				storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue1, columnValues2));

				//third computation
				columnValues = _mm_loadu_ps(A + i + (k + 2) * n);
				storedValues = _mm_add_ps(storedValues, _mm_mul_ps(transposeValue2, columnValues));

				columnValues2 = _mm_loadu_ps(A + (i + 4) + (k + 2) * n);
				storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue2, columnValues2));

				//fourth computation
				columnValues = _mm_loadu_ps(A + i + (k + 3) * n);
				storedValues = _mm_add_ps(storedValues, _mm_mul_ps(transposeValue3, columnValues));

				columnValues2 = _mm_loadu_ps(A + (i + 4) + (k + 3) * n);
				storedValues2 = _mm_add_ps(storedValues2, _mm_mul_ps(transposeValue3, columnValues2	));

				_mm_storeu_ps(position, storedValues);
				_mm_storeu_ps(position + 4, storedValues2);
			}
		}
	}
    
}



/*
    for( int j = 0; j < n; j++ ) {
    	for( int i = 0; i < (n - (n % 4)); i += 4 ){
    		__m128 newValue = _mm_loadu_ps (C + i+j*n);
    		for( int k = 0; k < (m - (m % 4)); k += 4 ) {
    			__m128 value2 = _mm_load1_ps (A + j*(n+1)+k*(n));
    			__m128 value1 = _mm_loadu_ps (A + i+k*n);
				__m128 intValue = _mm_mul_ps (value1, value2);
				newValue = _mm_add_ps (intValue, newValue);
				value2 = _mm_load1_ps (A + j*(n+1)+(k+1)*(n));
    			value1 = _mm_loadu_ps (A + i+(k+1)*n);
				intValue = _mm_mul_ps (value1, value2);
				newValue = _mm_add_ps (intValue, newValue);
				value2 = _mm_load1_ps (A + j*(n+1)+(k+2)*(n));
    			value1 = _mm_loadu_ps (A + i+(k+2)*n);
				intValue = _mm_mul_ps (value1, value2);
				newValue = _mm_add_ps (intValue, newValue);
				value2 = _mm_load1_ps (A + j*(n+1)+(k+3)*(n));
    			value1 = _mm_loadu_ps (A + i+(k+3)*n);
				intValue = _mm_mul_ps (value1, value2);
				newValue = _mm_add_ps (intValue, newValue);
    		}
    		for( int k = (m - (m % 4)); k < m; k += 1) {
    			__m128 value2 = _mm_load1_ps (A + j*(n+1)+k*(n));
    			__m128 value1 = _mm_loadu_ps (A + i+k*n);
				__m128 intValue = _mm_mul_ps (value1, value2);
				newValue = _mm_add_ps (intValue, newValue);
    		}
    		_mm_storeu_ps(C + i+j*n, newValue);
		}
    	for(int i = (n - (n % 4)); i < n; i++) {
    		float sum = C[i+j*n]; 
    		for(int k = 0; k < (m - (m % 4)); k += 4) {
    			sum += A[i+k*(n)] * A[j*(n+1)+k*(n)];
    			sum += A[i+(k+1)*(n)] * A[j*(n+1)+(k+1)*(n)];
    			sum += A[i+(k+2)*(n)] * A[j*(n+1)+(k+2)*(n)];
    			sum += A[i+(k+3)*(n)] * A[j*(n+1)+(k+3)*(n)];
    		}
    		for( int k = (m - (m % 4)); k < m; k += 1) {
    			sum += A[i+k*(n)] * A[j*(n+1)+k*(n)];
    		}
    		C[i+j*n] = sum;
    	}
	}*/

/* Naive operation. M is the size of the strip. N is the number of rows of A. 
   N+D is the number of colums of A. */
/*void sgemm( int m, int n, int d, float *A, float *C )
{
	for( int j = 0; j < n; j++ ) {
		for( int k = 0; k < m; k++ ) {
			__m128 value2 = _mm_load1_ps (A + j*(n+1)+k*(n));
			for( int i = 0; i < (n - (n % 16)); i += 16 ) {
				__m128 value1 = _mm_loadu_ps (A + i+k*n);
				__m128 intValue = _mm_mul_ps (value1, value2);
				__m128 oldValue = _mm_loadu_ps (C + i+j*n);
				__m128 newValue = _mm_add_ps (intValue, oldValue);
				_mm_storeu_ps(C + i+j*n, newValue);
				value1 = _mm_loadu_ps (A + (i+4)+k*n);
				intValue = _mm_mul_ps (value1, value2);
				oldValue = _mm_loadu_ps (C + (i+4)+j*n);
				newValue = _mm_add_ps (intValue, oldValue);
				_mm_storeu_ps(C + (i+4)+j*n, newValue);
				value1 = _mm_loadu_ps (A + (i+8)+k*n);
				intValue = _mm_mul_ps (value1, value2);
				oldValue = _mm_loadu_ps (C + (i+8)+j*n);
				newValue = _mm_add_ps (intValue, oldValue);
				_mm_storeu_ps(C + (i+8)+j*n, newValue);
				value1 = _mm_loadu_ps (A + (i+12)+k*n);
				intValue = _mm_mul_ps (value1, value2);
				oldValue = _mm_loadu_ps (C + (i+12)+j*n);
				newValue = _mm_add_ps (intValue, oldValue);
				_mm_storeu_ps(C + (i+12)+j*n, newValue);
			}
	      	for( int i = (n - (n % 16)); i < (n - (n % 4)); i += 4 ) {
				__m128 value1 = _mm_loadu_ps (A + i+k*n);
				__m128 intValue = _mm_mul_ps (value1, value2);
				__m128 oldValue = _mm_loadu_ps (C + i+j*n);
				__m128 newValue = _mm_add_ps (intValue, oldValue);
				_mm_storeu_ps(C + i+j*n, newValue);
			}
			for ( int i = (n - (n % 4)); i < n; i++) {
				C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
			}	
		}
	} 
}*/

// C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
/*void sgemm( int m, int n, int d, float *A, float *C )
{ 
	for (int k = 0; k < m; k++) {
		for (int j = 0; j < n; j +=4) {
			__m128 transposeValue0 = _mm_load1_ps(A + j*(n+1) + k * n);
			__m128 transposeValue1 = _mm_load1_ps(A + (j+1)*(n+1) + k * n);
			__m128 transposeValue2 = _mm_load1_ps(A + (j+2)*(n+1) + k * n);
			__m128 transposeValue3 = _mm_load1_ps(A + (j+3)*(n+1) + k * n);
			for (int i = 0; i < n; i += 4) {
				__m128 stripValues = _mm_loadu_ps(A + i + k * n);
				__m128 computeValue = _mm_loadu_ps(C + i + j * n);
				__m128 mulValues = _mm_mul_ps(stripValues, transposeValue0);
				computeValue = _mm_add_ps(computeValue, mulValues);
				_mm_storeu_ps(C + i + j * n, computeValue);

				stripValues = _mm_loadu_ps(A + i + k * n);
				computeValue = _mm_loadu_ps(C + i + (j+1) * n);
				mulValues = _mm_mul_ps(stripValues, transposeValue1);
				computeValue = _mm_add_ps(computeValue, mulValues);
				_mm_storeu_ps(C + i + (j+1) * n, computeValue);

				stripValues = _mm_loadu_ps(A + i + k * n);
				computeValue = _mm_loadu_ps(C + i + (j+2) * n);
				mulValues = _mm_mul_ps(stripValues, transposeValue2);
				computeValue = _mm_add_ps(computeValue, mulValues);
				_mm_storeu_ps(C + i + (j+2) * n, computeValue);

				stripValues = _mm_loadu_ps(A + i + k * n);
				computeValue = _mm_loadu_ps(C + i + (j+3) * n);
				mulValues = _mm_mul_ps(stripValues, transposeValue3);
				computeValue = _mm_add_ps(computeValue, mulValues);
				_mm_storeu_ps(C + i + (j+3) * n, computeValue);
			}
		}
	} */


// C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
/* s1: vectorizing with sse instructions. -- implemented
   s2: loop ordering -- implmented 
   s3: register blocking with local variable -- implemented
   s4: implmenting loop unrolling --
   s5: according to whether we have sufficient performance cache blocking -- 
   s6: compiler tricks? -- 
   */


/*ordering*/

/* Padding or edge cases? */
