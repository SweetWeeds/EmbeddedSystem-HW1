#include "user.h"

__host__
int compareArray(int *arr1, int *arr2, int size) {
    int diff = 0;
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            diff++;
        }
    }
    return diff;
}

__host__
void print2DArr(int *arr2d, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", arr2d[i*col + j]);
        }
        printf("\n");
    }
}
