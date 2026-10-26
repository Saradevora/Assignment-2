import numpy as np
import scipy.sparse as sp
import os

SIZES = [100, 500, 1000, 2000]
DENSITIES = [0.01, 0.05, 0.10, 0.20] # 1%, 5%, 10%, 20% de nonzeros
SEED = 42

def generate_sparse_datasets():
    os.makedirs("inputs_sparse", exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    for n in SIZES:
        for density in DENSITIES:
            # Generar matriz aleatoria dispersa usando scipy
            mat_a = sp.random(n, n, density=density, format='csr', dtype=np.float64, random_state=rng)
            mat_b = sp.random(n, n, density=density, format='csr', dtype=np.float64, random_state=rng)
            
            # Guardar en formato binario custom para leer en C o procesar
            # (Podemos almacenar row_ptr, col_idx y values)
            print(f"[*] Generado n={n}, densidad={density*100}% | NNZ(A)={mat_a.nnz}, NNZ(B)={mat_b.nnz}")

if __name__ == "__main__":
    generate_sparse_datasets()