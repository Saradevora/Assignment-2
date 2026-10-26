#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n;
    int nnz;
    int* row_ptr;
    int* col_ind;
    double* values;
} CSRMatrix;

void free_csr(CSRMatrix* mat) {
    if (mat) {
        free(mat->row_ptr);
        free(mat->col_ind);
        free(mat->values);
        free(mat);
    }
}

CSRMatrix* multiply_csr(const CSRMatrix* A, const CSRMatrix* B) {
    int n = A->n;
    
    int* row_ptr_c = (int*)calloc(n + 1, sizeof(int));
    if (!row_ptr_c) return NULL;

    double* work_values = (double*)calloc(n, sizeof(double));
    int* work_index = (int*)malloc(n * sizeof(int));
    int* marker = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) marker[i] = -1;

    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int row_nnz = 0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int a_col = A->col_ind[p];
            double a_val = A->values[p];

            for (int q = B->row_ptr[a_col]; q < B->row_ptr[a_col+1]; q++) {
                int b_col = B->col_ind[q];
                double b_val = B->values[q];

                if (marker[b_col] != i) {
                    marker[b_col] = i;
                    work_index[row_nnz++] = b_col;
                }
            }
        }
        total_nnz += row_nnz;
        row_ptr_c[i + 1] = total_nnz;
    }

    CSRMatrix* C = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    C->n = n;
    C->nnz = total_nnz;
    C->row_ptr = row_ptr_c;
    C->col_ind = (int*)malloc(total_nnz * sizeof(int));
    C->values = (double*)malloc(total_nnz * sizeof(double));

    int current_nnz = 0;
    for (int i = 0; i < n; i++) {
        int row_start = C->row_ptr[i];
        int row_end = C->row_ptr[i+1];

        int temp_count = 0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int a_col = A->col_ind[p];
            double a_val = A->values[p];

            for (int q = B->row_ptr[a_col]; q < B->row_ptr[a_col+1]; q++) {
                int b_col = B->col_ind[q];
                double b_val = B->values[q];

                if (marker[b_col] != i + 1000000) { // Nuevo marcador único por fila
                    marker[b_col] = i + 1000000;
                    work_index[temp_count++] = b_col;
                    work_values[b_col] = 0.0;
                }
                work_values[b_col] += a_val * b_val;
            }
        }

        for (int k = 0; k < temp_count; k++) {
            int col = work_index[k];
            C->col_ind[current_nnz] = col;
            C->values[current_nnz] = work_values[col];
            current_nnz++;
        }
    }

    free(work_values);
    free(work_index);
    free(marker);
    return C;
}

int main() {
    printf("[OK] Compilación y prueba base de CSR en C lista.\n");
    return 0;
}