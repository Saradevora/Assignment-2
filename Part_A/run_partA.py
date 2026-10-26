import os
import sys
import subprocess
import numpy as np
import pandas as pd
import psutil

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

SIZES = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
BLOCK_SIZES = [16, 32, 64, 128]
REPETITIONS = 5
EXE_NAME = "matrix_mult_optimized.exe"
SRC_NAME = "matrix_mult_optimized.c"

def compile_c():
    print("[*] Verificando ejecutable C...")
    if os.path.exists(EXE_NAME):
        print("[OK] El ejecutable ya existe en la carpeta. Omitiendo compilación.")
        return

    gcc_path = r"C:\w64devkit\bin\gcc.exe"
    if not os.path.exists(gcc_path):
        gcc_path = "gcc"  # Fallback por si acaso

    print(f"[*] Compilando con {gcc_path} (-O2)...")
    cmd = [gcc_path, "-O2", SRC_NAME, "-o", EXE_NAME]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[!] Error de compilación:\n", res.stderr)
        sys.exit(1)

def run_part_a():
    compile_c()
    os.makedirs("data", exist_ok=True)
    raw_results = []

    print("[*] Iniciando benchmarks de la Parte A...")

    for n in SIZES:
        bin_a = f"inputs/matrix_A_{n}.bin"
        bin_b = f"inputs/matrix_B_{n}.bin"

        if not (os.path.exists(bin_a) and os.path.exists(bin_b)):
            print(f"[!] Ficheros binarios para n={n} no encontrados en 'inputs/'. Saltando...")
            continue

        print(f"\n--- Tamaño n = {n} ---")

        methods_to_run = [
            ("Baseline_IJK", 0),
            ("LoopReordering_IKJ", 0)
        ] + [("CacheBlocking", b) for b in BLOCK_SIZES]

        for method, b_size in methods_to_run:
            cmd = [f"./{EXE_NAME}", str(n), method, str(b_size), str(REPETITIONS), bin_a, bin_b]
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p_util = psutil.Process(proc.pid)

            peak_ram_bytes = 0

            while proc.poll() is None:
                try:
                    mem_info = p_util.memory_info()
                    ram = getattr(mem_info, 'wset', mem_info.rss)
                    if ram > peak_ram_bytes:
                        peak_ram_bytes = ram
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

            stdout, stderr = proc.communicate()
            peak_ram_mib = peak_ram_bytes / (1024 * 1024)

            if proc.returncode != 0:
                print(f"[!] Error ejecutando {method} para n={n}:\n{stderr}")
                continue

            for line in stdout.strip().split("\n"):
                if line.startswith("REP,"):
                    parts = line.split(",")
                    rep_idx = int(parts[1])
                    elapsed_ms = float(parts[2])
                    checksum = float(parts[3])

                    label = f"{method}_B{b_size}" if method == "CacheBlocking" else method

                    raw_results.append({
                        "Size": n,
                        "Method": method,
                        "BlockSize": b_size,
                        "Label": label,
                        "Repetition": rep_idx,
                        "Time_ms": elapsed_ms,
                        "Peak_RAM_MiB": round(peak_ram_mib, 3),
                        "Checksum": checksum,
                        "DataType": "float64",
                        "Threads": 1
                    })

            times = [r["Time_ms"] for r in raw_results if r["Size"] == n and r["Label"] == (f"{method}_B{b_size}" if method == "CacheBlocking" else method)]
            med_t = np.median(times)
            iqr_t = np.percentile(times, 75) - np.percentile(times, 25)
            disp_name = f"CacheBlocking (B={b_size})" if method == "CacheBlocking" else method
            print(f"  -> {disp_name:<25}: Median = {med_t:8.4f} ms | IQR = {iqr_t:6.4f} ms | Peak RAM = {peak_ram_mib:6.2f} MiB")

    df_raw = pd.DataFrame(raw_results)
    raw_csv_path = "data/partA_dense_raw_results.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n[OK] Datos crudos guardados en '{raw_csv_path}'.")

if __name__ == "__main__":
    run_part_a()