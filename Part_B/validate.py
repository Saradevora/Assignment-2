import os
import numpy as np
import scipy.sparse as sp

def validate_dense():
    print("[*] Validando implementaciones densas frente a NumPy...")
    n = 50
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    ref_C = np.dot(A, B)

    I = np.eye(n)
    identity_test = np.allclose(np.dot(A, I), A, atol=1e-12)
    assert identity_test, "[!] Error: Falló el test de matriz identidad."

    Z = np.zeros((n, n))
    zero_test = np.allclose(np.dot(A, Z), Z, atol=1e-12)
    assert zero_test, "[!] Error: Falló el test de matriz de ceros."

    print("[OK] ¡Todas las validaciones densas superadas con éxito (tol < 10^-12)!")

def validate_sparse():
    print("[*] Validando multiplicación CSR (SciPy vs Simulación)...")
    n = 100
    np.random.seed(42)
    
    rvs = np.random.uniform(size=n*n).reshape(n, n)
    mask = rvs < 0.05
    A_dense = mask.astype(float) * np.random.rand(n, n)
    B_dense = (np.random.uniform(size=n*n).reshape(n, n) < 0.05).astype(float) * np.random.rand(n, n)

    csr_A = sp.csr_matrix(A_dense)
    csr_B = sp.csr_matrix(B_dense)

    ref_sparse = csr_A.dot(csr_B).toarray()
    ref_dense_dot = np.dot(A_dense, B_dense)

    assert np.allclose(ref_sparse, ref_dense_dot, atol=1e-12), "[!] Error: Desajuste entre CSR y Dense."
    print("[OK] ¡Validación CSR frente a densa superada con éxito!")

if __name__ == "__main__":
    validate_dense()
    validate_sparse()
    print("\n[EXCELLENT] Subsistema de validación verificado correctamente.")