"""
Numerical linear system solver module.

This module provides a generic linear system solver for solving systems
of the form A * x = b. It is independent of any aerodynamic or VLM-specific logic.

Classes:
    LinearSystemSolver: Solves linear systems using NumPy methods
"""

import numpy as np


class LinearSystemSolver:
    """
    Generic linear system solver.
    
    Solves linear systems of the form A * x = b using NumPy's linear algebra
    routines. Supports direct solving and least-squares methods, with optional
    condition number checking.
    
    Attributes:
        method (str): Solution method ('direct' or 'lstsq')
        check_condition (bool): Whether to compute condition number
    """
    
    def __init__(self, method="direct", check_condition=True):
        """
        Initialize the linear system solver.
        
        Parameters
        ----------
        method : str, optional
            Solution method. Supported values:
            - 'direct': Uses numpy.linalg.solve (default)
            - 'lstsq': Uses numpy.linalg.lstsq (least-squares)
        check_condition : bool, optional
            If True, compute the condition number of the matrix A.
            Default is True.
        """
        self.method = method
        self.check_condition = check_condition
    
    def solve(self, A, b):
        """
        Solve the linear system A * x = b.
        
        Parameters
        ----------
        A : numpy.ndarray
            Coefficient matrix, shape (N, N)
        b : numpy.ndarray
            Right-hand side vector, shape (N,) or (N, 1)
        
        Returns
        -------
        x : numpy.ndarray
            Solution vector, shape (N,)
        info : dict
            Dictionary containing solution information:
            - 'method': Solution method used
            - 'condition_number': Condition number of A (or None)
            - 'residual_norm': Norm of the residual A*x - b
        
        Raises
        ------
        ValueError
            If A is not a square matrix, if dimensions are incompatible,
            or if the solution method is unknown
        """
        # Validate input matrix A
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError("Matrix A must be 2-dimensional")
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        
        N = A.shape[0]
        
        # Validate and convert input vector b
        b = np.asarray(b)
        if b.ndim == 2:
            if b.shape[1] != 1:
                raise ValueError(f"Vector b has invalid shape {b.shape}")
            b = b.flatten()
        elif b.ndim != 1:
            raise ValueError(f"Vector b must be 1D or 2D, got {b.ndim}D")
        
        if b.shape[0] != N:
            raise ValueError(
                f"Incompatible dimensions: A is {A.shape}, b is {b.shape}"
            )
        
        # Compute condition number if requested
        condA = None
        if self.check_condition:
            condA = np.linalg.cond(A)
        
        # Solve the system based on the selected method
        if self.method == "direct":
            x = -np.linalg.solve(A, b)
        elif self.method == "lstsq":
            result = np.linalg.lstsq(A, b, rcond=None)
            x = -result[0]
        else:
            raise ValueError(f"Unknown solution method: '{self.method}'")
        
        # Compute residual norm
        r = A @ x - b
        residual_norm = np.linalg.norm(r)
        
        # Build info dictionary
        info = {
            "method": self.method,
            "condition_number": condA,
            "residual_norm": residual_norm
        }
        
        # Warn if matrix is ill-conditioned
        if condA is not None and condA > 1e10:
            print(f"Warning: Ill-conditioned matrix (condition number = {condA:.2e})")
        
        return x, info
