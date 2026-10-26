import numpy as np
import os

SIZES = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
SEED = 42

def generate_dense_inputs():
    os.makedirs("inputs", exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    print("[*] Generando ficheros binarios de matrices densas (seed=42)...")
    for n in SIZES:
        # Generar matrices con floats de 64 bits (double precision) en el rango [0, 1)
        matrix_a = rng.random((n, n), dtype=np.float64)
        matrix_b = rng.random((n, n), dtype=np.float64)
        
        # Guardar en formato binario plano Little-Endian (compatible con C)
        matrix_a.tofile(f"inputs/matrix_A_{n}.bin")
        matrix_b.tofile(f"inputs/matrix_B_{n}.bin")
        
    print("[OK] Todos los binarios densos se han generado correctamente en 'Part_A/inputs/'.")

if __name__ == "__main__":
    generate_dense_inputs()