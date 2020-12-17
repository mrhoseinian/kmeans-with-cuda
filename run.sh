nvcc -std=c++11 kmeans_cuda.cu -o kmeans_cuda -Wno-deprecated-gpu-targets
rm *.txt