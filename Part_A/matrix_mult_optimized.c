#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return ((double)count.QuadPart * 1000.0) / (double)freq.QuadPart;
}

void multiply_ijk(const double* restrict A, const double* restrict B, double* restrict C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiply_ikj(const double* restrict A, const double* restrict B, double* restrict C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

void multiply_blocking(const double* restrict A, const double* restrict B, double* restrict C, int n, int block_size) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += block_size) {
        int i_max = (ii + block_size < n) ? ii + block_size : n;
        for (int kk = 0; kk < n; kk += block_size) {
            int k_max = (kk + block_size < n) ? kk + block_size : n;
            for (int jj = 0; jj < n; jj += block_size) {
                int j_max = (jj + block_size < n) ? jj + block_size : n;

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i * n + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("Uso: %s <n> <method> <block_size> <reps> [file_A] [file_B]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    char* method = argv[2];
    int block_size = atoi(argv[3]);
    int reps = atoi(argv[4]);
    char* file_A = (argc >= 6) ? argv[5] : NULL;
    char* file_B = (argc >= 7) ? argv[6] : NULL;

    size_t matrix_size = (size_t)n * n;
    double* A = (double*)malloc(matrix_size * sizeof(double));
    double* B = (double*)malloc(matrix_size * sizeof(double));
    double* C = (double*)malloc(matrix_size * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error asignando memoria para n=%d\n", n);
        return 1;
    }

    if (file_A && file_B) {
        FILE* fA = fopen(file_A, "rb");
        FILE* fB = fopen(file_B, "rb");
        if (!fA || !fB) {
            fprintf(stderr, "Error abriendo ficheros binarios de entrada\n");
            return 1;
        }
        fread(A, sizeof(double), matrix_size, fA);
        fread(B, sizeof(double), matrix_size, fB);
        fclose(fA);
        fclose(fB);
    } else {
        for (size_t i = 0; i < matrix_size; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }

    for (int r = 0; r < reps; r++) {
        double t0 = get_time_ms();

        if (strcmp(method, "Baseline_IJK") == 0) {
            multiply_ijk(A, B, C, n);
        } else if (strcmp(method, "LoopReordering_IKJ") == 0) {
            multiply_ikj(A, B, C, n);
        } else if (strcmp(method, "CacheBlocking") == 0) {
            multiply_blocking(A, B, C, n, block_size);
        } else {
            fprintf(stderr, "Método desconocido: %s\n", method);
            free(A); free(B); free(C);
            return 1;
        }

        double t1 = get_time_ms();
        double elapsed = t1 - t0;

        double checksum = C[0] + C[matrix_size - 1];
        printf("REP,%d,%.4f,%.4f\n", r + 1, elapsed, checksum);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}