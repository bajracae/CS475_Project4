#include <stdio.h>
#include <stdlib.h> 
#include <xmmintrin.h>
#include <time.h>
#include <omp.h>


#define SSE_WIDTH		4

#ifndef NUMT
#define NUMT      1
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE      1000
#endif

#define NUM_ELEMENTS_PER_CORE   ARRAY_SIZE/NUMT

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	1000
#endif

float SimdMulSum( float *a, float *b, int len );
float nonSimdMulSum(float * a, float * b, int len);
float * generateArray(int length);

int main(){
    float * a = generateArray(ARRAY_SIZE);
    float * b = generateArray(ARRAY_SIZE);
    
    omp_set_num_threads(NUMT);
    
    float maxSimdPerformance = 0.0;
    float maxNonSimdPerformance = 0.0;

    
    for(int i = 0; i < NUMTRIES; i++){
        double time0 = omp_get_wtime();
        #pragma omp parallel
        {
            int first = omp_get_thread_num( ) * NUM_ELEMENTS_PER_CORE;
            SimdMulSum(&a[first], &b[first], NUM_ELEMENTS_PER_CORE);
        }
        double time1 = omp_get_wtime();
        
        double megaMulsPerSecond_Simd = (float)ARRAY_SIZE / (time1 - time0) / 1000000.0;
        if (megaMulsPerSecond_Simd > maxSimdPerformance){
			maxSimdPerformance = megaMulsPerSecond_Simd;
		}
        
        double time3 = omp_get_wtime();
        nonSimdMulSum(a, b, ARRAY_SIZE);
        double time4 = omp_get_wtime();
        
        double megaMulsPerSecond_nonSimd = (float)ARRAY_SIZE / (time4 - time3) / 1000000.0;
        if (megaMulsPerSecond_nonSimd > maxNonSimdPerformance){
        	maxNonSimdPerformance = megaMulsPerSecond_nonSimd;
        }
    }
    
    printf("%d,%d,%lf\n", NUMT, ARRAY_SIZE, maxSimdPerformance);
    printf("%d,%d,%lf\n", NUMT, ARRAY_SIZE, maxNonSimdPerformance);
    
    delete[] a;
    delete[] b;
    return 0;
}

float
SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;
    
	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

float nonSimdMulSum(float * a, float * b, int len){
    float sum = 0.;
    #pragma omp parallel for reduction(+:sum)
    for( int i = 0; i < len; i++){
        sum += a[i] * b[i];
    }
   return sum;
}

float * generateArray(int length){
    srand(time(0));
    
    float * random = (float *)malloc(sizeof(float) * length);
    
    for (int i = 0; i < length; i++) {
        random[i] = ((float)rand()/(float)(RAND_MAX)) * 100;
    }
    return random;
}
