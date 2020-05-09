#include <stdio.h>
#include <stdlib.h> 
#include <xmmintrin.h>
#include <time.h>
#include <omp.h>


#define SSE_WIDTH		4

#ifndef ARRAY_SIZE
#define ARRAY_SIZE      1000
#endif

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
    
    float maxSimdPerformance = 0.0;
    float maxNonSimdPerformance = 0.0;

    
    for(int i = 0; i < NUMTRIES; i++){
        double time0 = omp_get_wtime();
        float temp = SimdMulSum(a, b, ARRAY_SIZE);
        double time1 = omp_get_wtime();
        
        double time3 = omp_get_wtime();
        float temp_2 = nonSimdMulSum(a, b, ARRAY_SIZE);
        double time4 = omp_get_wtime();
        
        double megaMulsPerSecond_Simd = (float)ARRAY_SIZE / (time1 - time0) / 1000000.0;
        if (megaMulsPerSecond_Simd > maxSimdPerformance){
			maxSimdPerformance = megaMulsPerSecond_Simd;
		}
        
        double megaMulsPerSecond_nonSimd = (float)ARRAY_SIZE / (time4 - time3) / 1000000.0;
        if (megaMulsPerSecond_nonSimd > maxNonSimdPerformance){
			maxNonSimdPerformance = megaMulsPerSecond_nonSimd;
		}
    }

    // printf("%d\n", ARRAY_SIZE);
    // printf("Simd Performance: %lf\n", maxSimdPerformance);
    // printf("non-Simd Performance: %lf\n", maxNonSimdPerformance);
    printf("%d,%lf\n", ARRAY_SIZE, (maxSimdPerformance/maxNonSimdPerformance));
    // printf("======================================================\n");


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
    // printf("Simd: %lf\n", sum[0] + sum[1] + sum[2] + sum[3]);
	return sum[0] + sum[1] + sum[2] + sum[3];
}

float nonSimdMulSum(float * a, float * b, int len){
    float sum[4] = { 0., 0., 0., 0. };
    
    for( int i = 0; i < len; i++){
        sum[0] += a[i] * b[i];
    }
    
    // printf("Non-Simd: %lf\n", sum[0] + sum[1] + sum[2] + sum[3]);
    return sum[0] + sum[1] + sum[2] + sum[3];
}

float * generateArray(int length){
    srand(time(0));
    
    float * random = (float *)malloc(sizeof(float) * length);
    
    for (int i = 0; i < length; i++) {
        random[i] = ((float)rand()/(float)(RAND_MAX)) * 100;
    }
    return random;
}
