import os
import time
import numpy as np
import scipy.sparse as sp
import psutil
import csv

# Tamaños y niveles de densidad exigidos en el estudio
SIZES = [100, 500, 1000]
DENSITIES = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
REPETITIONS = 5
SEED = 42

def run_sparse_benchmark():
    results = []
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    for n in SIZES:
        for density in DENSITIES:
            print(f"[*] Evaluando tamaño n = {n} con densidad = {density*100}%...")
            
            # Generar matrices dispersas de prueba usando scipy (CSR format)
            mat_a_sp = sp.random(n, n, density=density, format='csr', dtype=np.float64, random_state=rng)
            mat_b_sp = sp.random(n, n, density=density, format='csr', dtype=np.float64, random_state=rng)
            
            # Convertir a densas equivalentes para la comparación justa exigida por el curso
            mat_a_dense = mat_a_sp.toarray()
            mat_b_dense = mat_b_sp.toarray()
            
            # --- 1. Benchmarking Método Denso ---
            dense_times = []
            dense_ram = []
            for _ in range(REPETITIONS):
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                start_time = time.perf_counter()
                # Multiplicación densa estándar
                _ = np.dot(mat_a_dense, mat_b_dense)
                end_time = time.perf_counter()
                
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                dense_times.append((end_time - start_time) * 1000.0) # en ms
                dense_ram.append(mem_after)
            
            # --- 2. Benchmarking Método Disperso (CSR) ---
            sparse_times = []
            sparse_ram = []
            for _ in range(REPETITIONS):
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                start_time = time.perf_counter()
                # Multiplicación dispersa CSR (usando el soporte eficiente de scipy)
                res_sp = mat_a_sp.dot(mat_b_sp)
                end_time = time.perf_counter()
                
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                sparse_times.append((end_time - start_time) * 1000.0) # en ms
                sparse_ram.append(mem_after)
            
            # Registrar resultados del método denso
            results.append({
                "Size": n,
                "Density": density,
                "Method": "Dense",
                "Median_Time_ms": np.median(dense_times),
                "IQR_Time_ms": np.percentile(dense_times, 75) - np.percentile(dense_times, 25),
                "Peak_RAM_MiB": np.median(dense_ram),
                "Output_NNZ": n * n  # En denso siempre es n^2
            })
            
            # Registrar resultados del método disperso
            results.append({
                "Size": n,
                "Density": density,
                "Method": "Sparse_CSR",
                "Median_Time_ms": np.median(sparse_times),
                "IQR_Time_ms": np.percentile(sparse_times, 75) - np.percentile(sparse_times, 25),
                "Peak_RAM_MiB": np.median(sparse_ram),
                "Output_NNZ": res_sp.nnz
            })
            
            print(f"    -> Dense: {np.median(dense_times):.3f} ms | Sparse CSR: {np.median(sparse_times):.3f} ms (NNZ saliente: {res_sp.nnz})")

    # Guardar en CSV para la memoria del informe
    csv_file = "data/partB_sparse_results.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["Size", "Density", "Method", "Median_Time_ms", "IQR_Time_ms", "Peak_RAM_MiB", "Output_NNZ"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[OK] Resultados de la Parte B guardados correctamente en '{csv_file}'.")

if __name__ == "__main__":
    run_sparse_benchmark()