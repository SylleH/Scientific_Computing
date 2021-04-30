#Project Scientific Computing, Master Applied Mathematics TU Delft
#Autors: N.J. Hulst and S. Hoogeveen

import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import math

alpha = 1
beta = 0


class SetupMatrices:
    def __init__(self, h=0.1, c=1, epsilon=10 ** -5):
        self.epsilon = epsilon
        self.h = h
        self.N = int(1 / self.h)
        self.c = c
        self.H_2 = 2 * self.h
        self.N_2h = math.floor(1 / self.H_2)

        self.A = np.zeros((self.N - 1, self.N - 1))
        self.f = np.zeros(self.N - 1)
        self.x = np.zeros(self.N - 1)
        self.u_ex = np.zeros(self.N - 1)

        self.I = np.identity(self.N - 1)

        self.setupProblem()
        self.setupDerivedMatrices()

    def setupProblem(self):
        for i in range(self.N - 1):
            self.x[i] = (i + 1) * self.h
            self.u_ex[i] = math.exp(self.x[i]) * (1 - self.x[i])
            self.f[i] = (1 + self.c) * math.exp(self.x[i]) + (1 - self.c) * self.x[i] * math.exp(self.x[i])
            if i == 0:
                self.f[i] = self.f[i] + alpha / self.h ** 2
            if i == self.N - 2:
                self.f[i] = self.f[i] + beta / self.h ** 2

        for i in range(self.N - 1):
            for j in range(self.N - 1):
                if j == i:
                    self.A[i][j] = 2 / self.h ** 2 + self.c
                if j == i - 1 or j == i + 1:
                    self.A[i][j] = -1 / self.h ** 2

    def setupDerivedMatrices(self):
        self.M_GS = np.tril(self.A, 0)  # is M1 in SGS
        self.M_GSB = np.triu(self.A, 0)  # is M2 in SGS
        D = np.triu(self.M_GS, 0)

        self.M_GS_inv = np.linalg.inv(self.M_GS)
        self.M_GSB_inv = np.linalg.inv(self.M_GSB)

        self.B_GS = self.I - self.M_GS_inv @ self.A
        self.B_GSB = self.I - self.M_GSB_inv @ self.A

        D_inv = np.linalg.inv(D)
        self.M_SGS = self.M_GSB @ D_inv @ self.M_GS
        self.M_SGS_inv = (self.M_GS_inv @ D) @ self.M_GSB_inv
        self.PO_SGS = self.M_SGS_inv @ self.A

        self.Ih_2h = np.zeros((self.N_2h, self.N - 1))  # restriction operator: half weighted
        self.Ih_2h[0][0] = 1 / 2
        self.Ih_2h[0][1] = 1 / 4
        self.Ih_2h[self.N_2h - 1][self.N - 3] = 1 / 4
        self.Ih_2h[self.N_2h - 1][self.N - 2] = 1 / 2
        for i in range(1, self.N_2h - 1):
            for j in range(self.N - 1):
                if j == 2 * i - 1:
                    self.Ih_2h[i][j] = 1 / 4
                    self.Ih_2h[i][j + 1] = 1 / 2
                    self.Ih_2h[i][j + 2] = 1 / 4

        self.I2h_h = 2 * (np.transpose(self.Ih_2h))  # prolongation operator: linear interpolation
        self.A_2h = self.Ih_2h @ (self.A @ self.I2h_h)  # coarse grid matrix obtained by Galerkin approach
        self.A_2h_inv = np.linalg.inv(self.A_2h)
        self.M_CGC_inv = self.I2h_h @ (self.A_2h_inv @ self.Ih_2h)
        self.B_CGC = self.I - self.M_CGC_inv @ self.A

        self.Proj = self.I - self.A @ self.M_CGC_inv

        self.B_TGM = self.B_GSB @ (self.B_CGC @ self.B_GS)
        self.M_TGM = self.A @ np.linalg.inv(self.I - self.B_TGM)
        self.M_TGM_inv = np.linalg.inv(self.M_TGM)

        self.PO_TGM = self.M_TGM_inv @ self.A


class NumSolve:
    def __init__(self, mat):
        pass

    def relError(self, mat):
        err_abs = norm(mat.u_ex - self.u)
        scale = norm(mat.u_ex)
        err_rel = err_abs / scale

        return err_abs, err_rel

    def properties(self, matrix):
        self.eigvals_full, _ = la.eig(matrix)
        self.eigvals_norm = [math.sqrt(self.eigvals_full.real[i] ** 2 + self.eigvals_full.imag[i] ** 2) for i in
                             range(len(self.eigvals_full))]
        self.rho = np.amax(self.eigvals_norm)  # spectral radius complex number

        eigvals_full_ATA, _ = la.eig((matrix.T) @ matrix)
        eigvals_norm_ATA = [math.sqrt(eigvals_full_ATA.real[i] ** 2 + eigvals_full_ATA.imag[i] ** 2) for i in
                            range(len(eigvals_full_ATA))]
        eigvals_max = np.amax(eigvals_norm_ATA)
        eigvals_min = np.amin(eigvals_norm_ATA)
        self.kappa = math.sqrt(eigvals_max / eigvals_min)  # condition number


class LU(NumSolve):
    def __init__(self, mat):
        # NumSolve.__init__(self, mat)

        self.LU = mat.A.copy()
        self.u = np.zeros(mat.N - 1)
        self.relError(mat)

    def LUdecomp(mat):
        L = np.zeros((mat.N - 1, mat.N - 1))

        for k in range(mat.N - 1):
            if self.LU[k][k] == 0:
                print('division by zero')
                return 0
            for i in range(k + 1, mat.N - 1):
                L[i][k] = self.LU[i][k] / self.LU[k][k]
                self.LU[i][k] = L[i][k]
                for j in range(k + 1, mat.N - 1):
                    self.LU[i][j] = self.LU[i][j] - L[i][k] * self.LU[k][j]

    def LUsolve(self):
        self.LUdecomp()

        y = np.zeros(mat.N - 1)
        L = np.tril(self.LU, -1) + mat.I
        U = np.triu(self.LU)

        for i in range(mat.N - 1):
            if i == 0:
                y[i] = mat.f[i].copy()
            else:
                y[i] = (mat.f[i] - np.dot(L[i][0:i], y[0:i]))

        for i in range(mat.N - 2, -1, -1):
            if i == mat.N - 2:
                self.u[i] = y[i] / U[i][i]
            else:
                self.u[i] = (y[i] - np.dot(U[i][i + 1:mat.N], self.u[i + 1:mat.N])) / U[i][i]


class GS(NumSolve):
    def __init__(self, mat):

        self.u = np.zeros(mat.N - 1)
        self.roc = 0
        self.roc_list = []
        self.relErr_list = []
        self.r_list = []

        self.relError(mat)
        self.properties(mat.B_GS)
        self.GSsolve(mat)

    def GSsolve(self, mat):
        # initialisation
        u_kMin1 = np.zeros(mat.N - 1)
        u_kMin2 = np.zeros(mat.N - 1)

        err = 2 * mat.epsilon
        k = 0

        while (err > mat.epsilon):  # and (k < 5000):
            self.u = mat.M_GS_inv @ ((mat.M_GS - mat.A) @ u_kMin1 + mat.f)

            r = mat.f - (mat.A @ self.u)
            err = norm(r) / norm(mat.f)
            _, rel_err = self.relError(mat)

            if k == 0 or k == 1:
                self.roc = 0
            else:
                self.roc = norm(self.u - u_kMin1) / norm(u_kMin1 - u_kMin2)

            self.r_list.append(r)
            self.relErr_list.append(rel_err)
            self.roc_list.append(self.roc)

            u_kMin2 = u_kMin1.copy()
            u_kMin1 = self.u.copy()

            k += 1


class CG(NumSolve):
    def __init__(self, mat, A, f, eps=False):

        self.u = np.zeros(mat.N - 1)
        self.roc = 0
        self.roc_list = []
        self.relErr_list = []
        self.r_list = []
        self.ritz_full = {}
        self.ritz_norm = {}
        if eps == False:
            self.epsilon = mat.epsilon
        else:
            self.epsilon = mat.epsilon * 10 * -1

        self.relError(mat)
        self.CGsolve(mat, A, f)

    def CGsolve(self, mat, A, f):

        # initialisation
        u_kMin1 = np.zeros(mat.N - 1)
        u_kMin2 = np.zeros(mat.N - 1)

        r_kMin1 = f.copy()

        err = 2 * self.epsilon
        k = 1

        R_k = np.array([r_kMin1 / norm(r_kMin1)]).T
        T_k = ((R_k.T) @ A) @ R_k
        ritz_full_k, _ = la.eig(T_k)
        ritz_norm_k = [math.sqrt(ritz_full_k.real[i] ** 2 + ritz_full_k.imag[i] ** 2) for i in
                       range(len(ritz_full_k.real))]
        self.ritz_full[0] = ritz_full_k
        self.ritz_norm[0] = ritz_norm_k

        # self.relErr_list.append(norm(r_kMin1) / norm(mat.f))

        while k < 500:
            if k == 1:
                p_k = r_kMin1
            else:
                # compute the search direction
                beta_k = np.inner(r_kMin1, r_kMin1) / np.inner(r_kMin2, r_kMin2)
                p_k = r_kMin1 + beta_k * p_kMin1

            Ap_k = A @ p_k
            alpha_k = np.inner(r_kMin1, r_kMin1) / np.inner(p_k, Ap_k)

            # update iterate and residual
            self.u = u_kMin1 + alpha_k * p_k
            r_k = r_kMin1 - alpha_k * (Ap_k)

            err = norm(r_k) / norm(mat.f)
            _, rel_err = self.relError(mat)

            if k == 1:
                self.roc = 0
            else:
                self.roc = norm(self.u - u_kMin1) / norm(u_kMin1 - u_kMin2)

            self.r_list.append(r_k)
            self.relErr_list.append(rel_err)
            self.roc_list.append(self.roc)

            if err < self.epsilon:
                break

            R_k = np.hstack((R_k, np.array([r_k / norm(r_k)]).T))
            T_k = ((R_k.T) @ A) @ R_k
            ritz_full_k, _ = la.eig(T_k)
            ritz_norm_k = [math.sqrt(ritz_full_k.real[i] ** 2 + ritz_full_k.imag[i] ** 2) for i in
                           range(len(ritz_full_k.real))]
            self.ritz_full[k] = ritz_full_k
            self.ritz_norm[k] = ritz_norm_k

            u_kMin2 = u_kMin1.copy()
            u_kMin1 = self.u.copy()
            r_kMin2 = r_kMin1.copy()
            r_kMin1 = r_k.copy()
            p_kMin1 = p_k.copy()

            k += 1


class CGC(NumSolve):
    def __init__(self, mat):

        self.u = np.zeros(mat.N - 1)
        self.roc = 0
        self.roc_list = []
        self.relErr_list = []
        self.r_list = []

        self.relError(mat)
        self.properties(mat.B_CGC)
        self.CGCsolve(mat)

    def CGCsolve(self, mat):

        # initialisation
        u_kMin1 = np.zeros(mat.N - 1)
        u_kMin2 = np.zeros(mat.N - 1)

        r = mat.f.copy()

        err = 2 * mat.epsilon
        k = 0

        while (err > mat.epsilon) and (k < 100):
            self.u = u_kMin1 + mat.I2h_h @ (mat.A_2h_inv @ mat.Ih_2h) @ r

            r = mat.f - (mat.A @ self.u)
            err = norm(r) / norm(mat.f)
            _, rel_err = self.relError(mat)

            if k == 0 or k == 1:
                self.roc = 0
            else:
                self.roc = norm(self.u - u_kMin1) / norm(u_kMin1 - u_kMin2)

            self.r_list.append(r)
            self.relErr_list.append(rel_err)
            self.roc_list.append(self.roc)

            u_kMin2 = u_kMin1.copy()
            u_kMin1 = self.u.copy()

            k += 1


class TGM(NumSolve):
    def __init__(self, mat):

        self.u = np.zeros(mat.N - 1)
        self.roc = 0
        self.roc_list = []
        self.relErr_list = []
        self.r_list = []

        self.relError(mat)
        self.properties(mat.B_TGM)
        self.TGMsolve(mat)

    def TGMsolve(self, mat):

        # initialisation
        u_kMin1 = np.zeros(mat.N - 1)
        u_kMin2 = np.zeros(mat.N - 1)

        err = 2 * mat.epsilon
        k = 0

        while (err > mat.epsilon) and (k < 5000):
            u_k1 = mat.B_GS @ u_kMin1 + mat.M_GS_inv @ mat.f
            r_h = mat.f - (mat.A @ u_k1)
            r_2h = mat.Ih_2h @ (r_h)
            e_2h = mat.A_2h_inv @ r_2h
            e_h = mat.I2h_h @ e_2h
            u_k2 = u_k1 + e_h
            self.u = mat.B_GSB @ u_k2 + mat.M_GSB_inv @ mat.f

            r = mat.f - (mat.A @ self.u)
            err = norm(r) / norm(mat.f)
            _, rel_err = self.relError(mat)

            if k == 0 or k == 1:
                self.roc = 0
            else:
                self.roc = norm(self.u - u_kMin1) / norm(u_kMin1 - u_kMin2)

            self.r_list.append(r)
            self.relErr_list.append(rel_err)
            self.roc_list.append(self.roc)

            u_kMin2 = u_kMin1.copy()
            u_kMin1 = self.u.copy()

            k += 1


