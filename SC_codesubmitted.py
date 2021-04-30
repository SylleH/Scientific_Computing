#Project Scientific Computing, Master Applied Mathematics TU Delft
#Autors: N.J. Hulst and S. Hoogeveen

from SC_class import SetupMatrices, NumSolve, LU, GS, CG, CGC, TGM
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def properties(A):
    eigvals_full, _ = la.eig(A)
    eigvals_norm = [math.sqrt(eigvals_full.real[i] ** 2 + eigvals_full.imag[i] ** 2) for i in range(len(eigvals_full))]
    rho = np.amax(eigvals_norm)  # spectral radius complex number

    eigvals_full_ATA, _ = la.eig((A.T) @ A)
    eigvals_norm_ATA = [math.sqrt(eigvals_full_ATA.real[i] ** 2 + eigvals_full_ATA.imag[i] ** 2) for i in
                        range(len(eigvals_full_ATA))]
    eigvals_max = np.amax(eigvals_norm_ATA)
    eigvals_min = np.amin(eigvals_norm_ATA)
    kappa = math.sqrt(eigvals_max / eigvals_min)  # condition number
    return eigvals_norm, rho, kappa


h_list = [0.1, 0.01, 0.005]
c_list = [-5, -1, 0, 1, 5]

fig_B_GS_eigs, ax_B_GS_eigs = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_GS, ax_u_GS = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_GS, ax_roc_GS = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_PCG_SGS, ax_u_PCG_SGS = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_PCG_SGS, ax_roc_PCG_SGS = plt.subplots(nrows=1, ncols=len(h_list))
# fig_PO_SGS_eigs_ritz, ax_PO_SGS_eigs_ritz = plt.subplots(nrows=1, ncols=len(h_list))
fig_B_CGC_eigs, ax_B_CGC_eigs = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_CGC, ax_u_CGC = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_CGC, ax_roc_CGC = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_CG_proj, ax_u_CG_proj = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_CG_proj, ax_roc_CG_proj = plt.subplots(nrows=1, ncols=len(h_list))
fig_B_TGM_eigs, ax_B_TGM_eigs = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_TGM, ax_u_TGM = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_TGM, ax_roc_TGM = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_PCG_TGM, ax_u_PCG_TGM = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_PCG_TGM, ax_roc_PCG_TGM = plt.subplots(nrows=1, ncols=len(h_list))
# fig_PO_TGM_eigs_ritz, ax_PO_TGM_eigs_ritz = plt.subplots(nrows=1, ncols=len(h_list))
fig_u_CG, ax_u_CG = plt.subplots(nrows=1, ncols=len(h_list))
fig_roc_CG, ax_roc_CG = plt.subplots(nrows=1, ncols=len(h_list))

rho_GS_dict = {}
roc_GS_dict = {}
kappa_PO_SGS_dict = {}
rho_CGC_dict = {}
roc_CGC_dict = {}
rho_TGM_dict = {}
roc_TGM_dict = {}
kappa_PO_TGM_dict = {}

for i in range(len(h_list)):
    rho_GS_in = {}
    roc_GS_in = {}
    kappa_PO_SGS_in = {}
    rho_CGC_in = {}
    roc_CGC_in = {}
    rho_TGM_in = {}
    roc_TGM_in = {}
    kappa_PO_TGM_in = {}

    for j in range(len(c_list)):
        mat = SetupMatrices(h=h_list[i], c=c_list[j])
        solLU = LU(mat)
        solGS = GS(mat)
        solCG = CG(mat, mat.A, mat.f)
        solCG_proj = CG(mat, mat.Proj @ mat.A, mat.Proj @ mat.f)
        solPCG_SGS = CG(mat, mat.M_SGS_inv @ mat.A, mat.M_SGS_inv @ mat.f)
        solCGC = CGC(mat)
        solTGM = TGM(mat)
        solPCG_TGM = CG(mat, mat.M_TGM_inv @ mat.A, mat.M_TGM_inv @ mat.f)

        ax_B_GS_eigs[i].plot(np.sort(solGS.eigvals_norm), 'o', markersize=6, label='c = ' + str(c_list[j]))
        ax_u_GS[i].plot(mat.x, solGS.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_GS[i].plot([math.log(i) for i in solGS.relErr_list], '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_u_PCG_SGS[i].plot(mat.x, solPCG_SGS.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_PCG_SGS[i].plot([math.log(i) for i in solPCG_SGS.relErr_list], '-', linewidth=2,
                               label='c = ' + str(c_list[j]))
        ax_B_CGC_eigs[i].plot(np.sort(solCGC.eigvals_norm), 'o', markersize=6, label='c = ' + str(c_list[j]))
        if i == 0:
            ax_u_CGC[i].plot(mat.x, solCGC.u, '-', linewidth=2, marker='o', markersize=6, label='c = ' + str(c_list[j]))
        else:
            ax_u_CGC[i].plot(mat.x, solCGC.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_CGC[i].plot([math.log(i) for i in solCGC.relErr_list], '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_u_CG_proj[i].plot(mat.x, solCG_proj.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_CG_proj[i].plot([math.log(i) for i in solCG_proj.relErr_list], '-', linewidth=2,
                               label='c = ' + str(c_list[j]))
        ax_B_TGM_eigs[i].plot(np.sort(solTGM.eigvals_norm), 'o', markersize=6, label='c = ' + str(c_list[j]))
        ax_u_TGM[i].plot(mat.x, solTGM.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_TGM[i].plot([math.log(i) for i in solTGM.relErr_list], '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_u_PCG_TGM[i].plot(mat.x, solPCG_TGM.u, label='c = ' + str(c_list[j]))
        ax_roc_PCG_TGM[i].plot([math.log(i) for i in solPCG_TGM.relErr_list], label='c = ' + str(c_list[j]))
        ax_u_CG[i].plot(mat.x, solCG.u, '-', linewidth=2, label='c = ' + str(c_list[j]))
        ax_roc_CG[i].plot([math.log(i) for i in solCG.relErr_list], '-', linewidth=2, label='c = ' + str(c_list[j]))

        c = c_list[j]

        rho_GS_in[c] = solGS.rho
        roc_GS_in[c] = solGS.roc
        _, _, kappa_PO_SGS_in[c] = properties(mat.PO_SGS)
        rho_CGC_in[c] = solCGC.rho
        roc_CGC_in[c] = solCGC.roc
        rho_TGM_in[c] = solTGM.rho
        roc_TGM_in[c] = solTGM.roc
        _, _, kappa_PO_TGM_in[c] = properties(mat.PO_TGM)

    h = h_list[i]

    rho_GS_dict[h] = rho_GS_in
    roc_GS_dict[h] = roc_GS_in
    kappa_PO_SGS_dict[h] = kappa_PO_SGS_in
    rho_CGC_dict[h] = rho_CGC_in
    roc_CGC_dict[h] = roc_CGC_in
    rho_TGM_dict[h] = rho_TGM_in
    roc_TGM_dict[h] = roc_TGM_in
    kappa_PO_TGM_dict[h] = kappa_PO_TGM_in

fig_list = [fig_B_GS_eigs, fig_u_GS, fig_roc_GS, fig_u_PCG_SGS, fig_roc_PCG_SGS,
            fig_B_CGC_eigs, fig_u_CGC, fig_roc_CGC, fig_u_CG_proj,
            fig_roc_CG_proj, fig_B_TGM_eigs, fig_u_TGM, fig_roc_TGM, fig_u_PCG_TGM,
            fig_roc_PCG_TGM, fig_u_CG, fig_roc_CG]

ax_list = [ax_B_GS_eigs, ax_u_GS, ax_roc_GS, ax_u_PCG_SGS, ax_roc_PCG_SGS,
           ax_B_CGC_eigs, ax_u_CGC, ax_roc_CGC, ax_u_CG_proj,
           ax_roc_CG_proj, ax_B_TGM_eigs, ax_u_TGM, ax_roc_TGM, ax_u_PCG_TGM,
           ax_roc_PCG_TGM, ax_u_CG, ax_roc_CG]

names_list = ["B_GS_eigs", "u_GS", "roc_GS", "u_PCG_SGS", "roc_PCG_SGS",
              "B_CGC_eigs", "u_CGC", "roc_CGC", "u_CG_proj",
              "roc_CG_proj", "B_TGM_eigs", "u_TGM", "roc_TGM", "u_PCG_TGM",
              "roc_PCG_TGM", "u_CG", "roc_CG"]

title_list = ["Eigenvalues $B_{GS}$", "Solution GS solver", "Convergence GS solver",
              "Solution CG with SGS preconditioner", "Convergence CG with SGS preconditioner",
              "Eigenvalues $B_{CGC}$",
              "Solution CGC solver", "Convergence CGC solver", "Solution CG projected system",
              "Convergence CG projected system", "Eigenvalues $B_{TGM}$", "Solution TGM solver",
              "Convergence TGM solver", "Solution CG with TGM preconditioner",
              "Convergence CG with TGM preconditioner",
              "Solution CG solver", "Convergence CG solver"]

for m in range(len(names_list)):
    fig = fig_list[m]
    ax = ax_list[m]
    name = names_list[m]

    fig.suptitle(title_list[m], fontsize=20)

    for i in range(len(h_list)):
        fig.tight_layout(h_pad=2)
        ax[i].title.set_text('h = ' + str(h_list[i]))

        if "u" in name:
            ax[i].plot(mat.x, mat.u_ex, '--', linewidth=2, color='k', label='exact')
            ax[i].set_xlabel("x")
            if i == 0:
                ax[i].set_ylabel("u")

        if "roc" in name:
            ax[i].set_xlabel("iteration number")
            if i == 0:
                ax[i].set_ylabel("$log||u_k - u||$")

        if "eig" in name:
            ax[i].set_xlabel("# of eigenvalue")
            if i == 0:
                ax[i].set_ylabel("norm of eigenvalue")

        ax[0].legend()
    fig.savefig(name + '.png')


def plot_lambda_and_ritz_norm(lambda_to_plot, ritz_to_plot, tit=None, xlab=None, ylab=None, leg=None, saveName=None):
    its = len(ritz_to_plot.keys())
    x_lambda = [its] * len(lambda_to_plot)

    plt.figure()
    plt.plot(x_lambda, lambda_to_plot, 'o', markersize=6, color='tab:orange')

    for it in range(0, its):
        x_ritz = [it] * len(ritz_to_plot[it])
        plt.plot(x_ritz, ritz_to_plot[it], 'o', markersize=6, color='tab:blue')

    if xlab != None:
        plt.xlabel(xlab)
    if ylab != None:
        plt.ylabel(ylab)

    plt.legend(['eigenvalues', 'Ritz values'])

    if tit != None:
        plt.title(tit)
    if saveName != None:
        plt.savefig(saveName + '.png')


h_list = [0.1, 0.01]
c_list = [-5, 5]

for s in [0, 1]:
    hh = h_list[s]
    cc = c_list[s]

    mat = SetupMatrices(h=hh, c=cc)
    solCG = CG(mat, mat.A, mat.f)
    solPCG_SGS = CG(mat, mat.M_SGS_inv @ mat.A, mat.M_SGS_inv @ mat.f)
    solPCG_TGM = CG(mat, mat.M_TGM_inv @ mat.A, mat.M_TGM_inv @ mat.f)

    eigvals_norm, _, _ = properties(mat.PO_SGS)
    plot_lambda_and_ritz_norm(eigvals_norm, solPCG_SGS.ritz_norm,
                              tit="Ritz values and eigenvalues SGS preconditioner (c = " + str(cc) + ", h = " + str(
                                  hh) + ")",
                              xlab='iteration nr',
                              ylab='value',
                              # saveName = 'eigsAndRitz_PO_SGS_c_'+str(c)+'_h_'+str(h))
                              saveName='eigsAndRitz_PO_SGS_' + str(s))

    eigvals_norm, _, _ = properties(mat.PO_TGM)
    plot_lambda_and_ritz_norm(eigvals_norm, solPCG_TGM.ritz_norm,
                              tit="Ritz values and eigenvalues TGM preconditioner (c = " + str(cc) + ", h = " + str(
                                  hh) + ")",
                              xlab='iteration nr',
                              ylab='value',
                              # saveName = 'eigsAndRitz_PO_TGM_c_'+str(c)+'_h_'+str(h))
                              saveName='eigsAndRitz_PO_TGM_' + str(s))

    eigvals_norm, _, _ = properties(mat.A)
    plot_lambda_and_ritz_norm(eigvals_norm, solCG.ritz_norm,
                              tit="Ritz values and eigenvalues A (c = " + str(cc) + ", h = " + str(hh) + ")",
                              xlab='iteration nr',
                              ylab='value',
                              # saveName = 'eigsAndRitz_CG_c_'+str(c)+'_h_'+str(h))
                              saveName='eigsAndRitz_PO_A_' + str(s))


def tabulate_for_c_and_h(dict_to_tabulate):
    temp_dict = {}
    for h in dict_to_tabulate.keys():
        temp = [dict_to_tabulate[h][c] for c in dict_to_tabulate[h]]
        temp_dict[h] = temp
    return pd.DataFrame(temp_dict)


rho_GS_tab = tabulate_for_c_and_h(rho_GS_dict)
rho_GS_tab.to_csv('rho_GS.csv', index=True)

u_GS_conv_tab = tabulate_for_c_and_h(roc_GS_dict)
u_GS_conv_tab.to_csv('roc_GS.csv', index=True)

kappa_PO_SGS_tab = tabulate_for_c_and_h(kappa_PO_SGS_dict)
kappa_PO_SGS_tab.to_csv('kappa_PO_SGS.csv', index=True)

rho_CGC_tab = tabulate_for_c_and_h(rho_CGC_dict)
rho_CGC_tab.to_csv('rho_CGC.csv', index=True)

u_CGC_conv_tab = tabulate_for_c_and_h(roc_CGC_dict)
u_CGC_conv_tab.to_csv('roc_CGC.csv', index=True)

rho_TGM_tab = tabulate_for_c_and_h(rho_TGM_dict)
rho_TGM_tab.to_csv('rho_TGM.csv', index=True)

u_TGM_conv_tab = tabulate_for_c_and_h(roc_TGM_dict)
u_TGM_conv_tab.to_csv('roc_TGM.csv', index=True)

kappa_PO_TGM_tab = tabulate_for_c_and_h(kappa_PO_TGM_dict)
kappa_PO_TGM_tab.to_csv('kappa_PO_TGM.csv', index=True)
