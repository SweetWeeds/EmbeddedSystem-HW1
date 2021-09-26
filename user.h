/** Start of "user.h" **/
#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#define MAT1_ROW    512
#define MAT1_COL    1024
#define MAT2_ROW    512
#define MAT2_COL    512
#define MATR_ROW    MAT1_ROW
#define MATR_COL    (MAT1_COL+MAT2_COL)

__host__ int compareArray(int *arr1, int *arr2, int size);
__host__ void print2DArr(int *arr2d, int row, int col);

#endif
/** End of "user.h" **/
