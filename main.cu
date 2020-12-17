#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>
#include <chrono>
#include <time.h>
using namespace std;
double passed_time;

__global__ void distance(float *dst, float *x, float *y,
                         float *mu_x, float *mu_y){
    int i = blockIdx.x;
    int j = threadIdx.x;

    dst[i * blockDim.x + j] = (x[i] - mu_x[j]) * (x[i] - mu_x[j]);
    dst[i * blockDim.x + j] += (y[i] - mu_y[j]) * (y[i] - mu_y[j]);
}

__global__ void clustering(int *group, float *dst, int k){
    int i = blockIdx.x;
    int j;
    float min_dst;

    min_dst = dst[i * k + 0];
    group[i] = 1;

    for(j = 1; j < k; ++j){
        if(dst[i * k + j] < min_dst){
            min_dst = dst[i * k + j];
            group[i] = j + 1;
        }
    }
}

__global__ void clear(float *sum_x, float *sum_y, int *nx, int *ny){
    int j = threadIdx.x;

    sum_x[j] = 0;
    sum_y[j] = 0;
    nx[j] = 0;
    ny[j] = 0;
}

__global__ void move_centroid(float *sum_x, float *sum_y, int *nx, int *ny,
                              float *x, float *y, int *group, int num_points){
    int i;
    int j = threadIdx.x;

    for(i = 0; i < num_points; ++i){
        if(group[i] == (j + 1)){
            sum_x[j] += x[i];
            sum_y[j] += y[i];
            nx[j]++;
            ny[j]++;
        }
    }
}

__global__ void move_mu(float *mu_x, float *mu_y, float *sum_x,
                        float *sum_y, int *nx, int *ny){
    int j = threadIdx.x;

    mu_x[j] = sum_x[j]/nx[j];
    mu_y[j] = sum_y[j]/ny[j];
}

void kmeans-kernel(int num_reps, int num_points, int k,
                   float *x_d, float *y_d, float *mu_x_d, float *mu_y_d,
                   int *group_d, int *nx_d, int *ny_d,
                   float *sum_x_d, float *sum_y_d, float *dst_d){
    int i;
    for(i = 0; i < num_reps; ++i){
        distance<<<num_points,k>>>(dst_d, x_d, y_d, mu_x_d, mu_y_d);
        clustering<<<num_points,1>>>(group_d, dst_d, k);
        clear<<<1,k>>>(sum_x_d, sum_y_d, nx_d, ny_d);
        move_centroid<<<1,k>>>(sum_x_d, sum_y_d, nx_d, ny_d, x_d, y_d, group_d, num_points);
        move_mu<<<1,k>>>(mu_x_d, mu_y_d, sum_x_d, sum_y_d, nx_d, ny_d);
    }
}


void read_samples(float **x, float **y, float **mu_x, float **mu_y, int *num_points, int *k,char* arg){
    FILE *fp;
    char buf[64];

    *num_points = 0;
    fp = fopen(arg, "r");

    while(fgets(buf, 64, fp) != NULL){
        *num_points += 1;
        *x = (float*) realloc(*x, (*num_points)*sizeof(float));
        *y = (float*) realloc(*y, (*num_points)*sizeof(float));
        istringstream line_stream(buf);
        float x1,y1;
        line_stream >> x1 >> y1;
        (*x)[*num_points - 1] = x1;
        (*y)[*num_points - 1] = y1;
    }
    fclose(fp);


    *k = 0;
    fp = fopen("input/initCoord.txt", "r");
    while(fgets(buf, 64, fp) != NULL){
        *k += 1;
        *mu_x = (float*) realloc(*mu_x, (*k)*sizeof(float));
        *mu_y = (float*) realloc(*mu_y, (*k)*sizeof(float));
        istringstream line_stream(buf);
        float x1,y1;
        line_stream >> x1 >> y1;
        (*mu_x)[*k - 1] = x1;
        (*mu_y)[*k - 1] = x1;
    }
    fclose(fp);
}


void verify(int *group, float *mu_x, float *mu_y, int num_points, int k,char* arg){
    FILE *fp;
    int i;
    string str(arg),str1,str2;
    str = "output/cuda/" + str;

    str1 = str + "_group_members.txt";
    fp = fopen(str1.c_str(), "w");
    for(i = 0; i < num_points; ++i){
        fprintf(fp, "%d\n", group[i]);
    }
    fclose(fp);

    str2 = str + "_centroids.txt";
    fp = fopen(str2.c_str(), "w");
    for(i = 0; i < k; ++i){
        fprintf(fp, "%0.6f %0.6f\n", mu_x[i], mu_y[i]);
    }
    fclose(fp);

    fp = fopen("CUDAtimes.txt", "a");
    fprintf(fp, "%0.6f\n", passed_time);
    fclose(fp);
}
int main(int argc,char* argv[]){

    // Initialize host variables ----------------------------------------------
    int num_points; /* number of points */
    int k; /* number of clusters */
    int *group;
    float *x = NULL, *y = NULL, *mu_x = NULL, *mu_y = NULL;

    // Initialize device variables --------------------------------------------
    int *group_d, *nx_d, *ny_d;
    float *x_d, *y_d, *mu_x_d, *mu_y_d, *sum_x_d, *sum_y_d, *dst_d;

    /* read data from files on cpu */
    read_samples(&x, &y, &mu_x, &mu_y, &num_points, &k,argv[2]);

    // Allocate host memory ----=====------------------------------------------
    group = (int*) malloc(num_points*sizeof(int));

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**) &group_d,num_points*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &nx_d, k*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &ny_d, k*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &x_d, num_points*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &y_d, num_points*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &mu_x_d, k*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &mu_y_d, k*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &sum_x_d, k*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &sum_y_d, k*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &dst_d, num_points*k*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(x_d, x, num_points*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(y_d, y, num_points*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(mu_x_d, mu_x, k*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(mu_y_d, mu_y, k*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);

    const auto start = chrono::high_resolution_clock::now();
    kmeans-kernel(100, num_points, k, x_d, y_d, mu_x_d, mu_y_d, group_d, nx_d, ny_d, sum_x_d, sum_y_d, dst_d);

    const auto end = chrono::high_resolution_clock::now();
    const auto duration =
            chrono::duration_cast<chrono::duration<float>>(end - start);
    cerr << "CUDA Took: " << duration.count() << "s" << " for "<<argv[3]<<" points." << endl;
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    passed_time = duration.count();

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(group, group_d, num_points*sizeof(int), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cuda_ret = cudaMemcpy(mu_x, mu_x_d, k*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cuda_ret = cudaMemcpy(mu_y, mu_y_d, k*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    // Verify correctness -----------------------------------------------------
    verify(group, mu_x, mu_y, num_points, k,argv[3]);

    // Free memory ------------------------------------------------------------
    free(x); free(y); free(mu_x); free(mu_y); free(group);

    cudaFree(x_d); cudaFree(y_d); cudaFree(mu_x_d); cudaFree(mu_y_d); cudaFree(group_d);
    cudaFree(nx_d); cudaFree(ny_d); cudaFree(sum_x_d); cudaFree(sum_y_d); cudaFree(dst_d);

    return 0;
}