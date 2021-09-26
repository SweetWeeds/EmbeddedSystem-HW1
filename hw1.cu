#include "user.h"
#include "user_host.h"
#include "user_device.h"

using namespace std;

int main(int argc, char *argv[]) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(1, 10);
    cudaEvent_t cuda_start, cuda_end;
    float host_exec_time, device_exec_time;
    
    int mat1_row = MAT1_ROW;
    int mat1_col = MAT1_COL;
    int mat2_row = MAT2_ROW;
    int mat2_col = MAT2_COL;
    int matr_row = MATR_ROW;
    int matr_col = MATR_COL;

    printf("mat1[%d][%d]\n", mat1_row, mat1_col);
    printf("mat2[%d][%d]\n", mat2_row, mat2_col);
    printf("matr[%d][%d]\n", matr_row, matr_col);

    int mat1_size = mat1_row * mat1_col * sizeof(int);
    int mat2_size = mat2_row * mat2_col * sizeof(int);
    int matr_size = matr_row * matr_col * sizeof(int);
    
    int *host_mat1 = NULL;
    int *host_mat2 = NULL;
    int *host_matr = NULL;
    int *device_mat1 = NULL;
    int *device_mat2 = NULL;
    int *device_matr = NULL;
    int *host_device_matr = NULL;
    
    // Start of Memory Allocation //
    host_mat1 = (int *)malloc(mat1_size);
    host_mat2 = (int *)malloc(mat2_size);
    host_matr = (int *)malloc(matr_size);
    host_device_matr = (int *)malloc(matr_size);
    cudaMalloc((void **)&device_mat1, mat1_size);
    cudaMalloc((void **)&device_mat2, mat2_size);
    cudaMalloc((void **)&device_matr, matr_size);
    // End of Memory Allocation //

    // Start of Array Initalization //
    // Matrix 1
    for (int i = 0; i < mat1_row; i++) {
        for (int j = 0; j < mat1_col; j++) {
            host_mat1[i * mat1_col + j] = dis(gen);
        }
    }

    // Matrix 2
    for (int i = 0; i < mat2_row; i++) {
        for (int j = 0; j < mat2_col; j++) {
            host_mat2[i * mat2_col + j] = dis(gen);
        }
    }
    // End of Array Initalization //

    // Memory Copy
    cudaMemcpy(device_mat1, host_mat1, mat1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat2, host_mat2, mat2_size, cudaMemcpyHostToDevice);

    // Prepare File Stream
    ofstream resultFile;
    resultFile.open("./result.csv", ios::out);
    resultFile << "Target,#Threads,#ThreadBlks,ExecTime\n";

    // Start of Concatenation (host) //
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    host_Concatenate(host_mat1, host_mat2, host_matr, &host_exec_time);
    gettimeofday(&endTime, NULL);
    host_exec_time = (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;
    resultFile << "host,1,1," << host_exec_time << "\n";
    printf("[INFO] Host Execution time:%lf\n", host_exec_time);
    // End of Concatenation (host) //

    int numElements = matr_col * matr_row;
    int numBlocks, numThreadsperBlock;
    for (numThreadsperBlock = NUM_THREADS_BASE; numThreadsperBlock <= NUM_THREADSA_MAX; numThreadsperBlock *= 2) {
        for (numBlocks = NUM_THREAD_BLKS_FROM; numBlocks <= NUM_THREAD_BLKS_TO; numBlocks *= 2) {
            int numOps = numElements > (numBlocks * numThreadsperBlock) ? numElements / (numBlocks * numThreadsperBlock) + (numElements % (numBlocks * numThreadsperBlock) ? 1 : 0) : 1;
            dim3 gridSize(numBlocks);
            dim3 blockSize(numThreadsperBlock);

            cudaMemset(device_matr, 0, numElements);
            cudaEventCreate(&cuda_start);
            cudaEventCreate(&cuda_end);
            cudaEventRecord(cuda_start, 0);
            device_Concatenate<<<gridSize, blockSize>>>(device_mat1, device_mat2, device_matr, numOps, numElements, mat1_col, mat2_col, matr_col);
            cudaEventRecord(cuda_end, 0);
            cudaEventSynchronize(cuda_end);
            cudaEventElapsedTime(&device_exec_time, cuda_start, cuda_end);
            cudaEventDestroy(cuda_start);
            cudaEventDestroy(cuda_end);

            resultFile << "device," << numThreadsperBlock << "," << numBlocks << "," << device_exec_time << "\n";
            cudaMemcpy(host_device_matr, device_matr, matr_size, cudaMemcpyDeviceToHost);
            int diff = compareArray(host_matr, host_device_matr, numElements);
            if (diff) {
                printf("[WARNING] ");
            } else {
                printf("[INFO] ");
            }
            printf("numOps: %d, numBlocks: %d, numThreadsperBlock: %d, diff: %d, exec_time: %.4lf\n", numOps, numBlocks, numThreadsperBlock, diff, device_exec_time);
        }
    }

    // Close File Stream
    resultFile.close();

    // End of Concatenation (device) //

    return 0;
}
