#include "user.h"

__host__
void host_Concatenate(int *host_mat1, int *host_mat2, int *host_matr, float *p_exec_time) {
    for (int i = 0; i < MAT1_ROW; i++) {
        for (int j = 0; j < MAT1_COL; j++) {
            host_matr[i * MATR_COL + j] = host_mat1[i * MAT1_COL + j];
        }
    }
    for (int i = 0; i < MAT2_ROW; i++) {
        for (int j = 0; j < MAT2_COL; j++) {
            host_matr[i * MATR_COL + MAT1_COL + j] = host_mat2[i * MAT2_COL + j];
        }
    }
}