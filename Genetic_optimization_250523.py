import sympy as sym
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pickle
#from plotta_sim_dep_ini import plot_tr_v3_vect3_dep_ini2, plot_tr_from_fit_neuron_dep_ini2
from load_eq import load_v3
from scipy.optimize import Bounds
from carica_exp_info import neuron_info_load
from neuron_model_utility import calcola_spikes_al_variare_Iadap, calcola_primi_spike_modello,calcola_monod
import json


def runsimulation(neuron_nb):
    def monod(x, a, b, c, alp):
        return c + (a * np.exp(b) * x) / (alp + x)

    def monod_tot(x):
        err = []
        for i in range(Istm.__len__()):
            if is_spk_2_curr[i]:
                #print(i)
                err.append(sum((x[2] + ((x[0] * np.exp(x[1] * Istm[i] / 1000).astype(np.float64) * np.array( t_sp_abs_tutti[i]).astype(np.float64))) / (x[3] + np.array(t_sp_abs_tutti[i]).astype(np.float64)) - np.array(Iada_tutti[i]).astype(np.float64)) ** 2))  # *np.array(t_var_tutti[corr_sp2-i-1]).astype(np.float64)))
        err_tot = sum(err)
        return err_tot.astype(np.float64)

    def calcola_Idep_2(Idep_ini_hypotesis, Idep_range, k, tol, Istm, sc, bet, delta1, psi1, Ith, time_scale):

        aux = Istm / sc
        alpha3 = aux.tolist()
        t = sym.Symbol('t')
        delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')
        [V, Iadap, Idep] = load_v3()
        n_corr=Istm.__len__()
        time_sp_hyp = np.zeros(n_corr)
        time_sp_hyp_inf = np.zeros(n_corr)
        time_sp_hyp_sup = np.zeros(n_corr)
        while k > 0:
            err_fs_hyp = np.zeros(n_corr)
            err_fs_hyp_inf = np.zeros(n_corr)
            err_fs_hyp_sup = np.zeros(n_corr)
            for i in range(n_corr):
                if is_spk_curr[i]:

                    Idep_ini_hypotesis_sup = Idep_ini_hypotesis + Idep_range
                    Idep_ini_hypotesis_inf = max(Idep_ini_hypotesis - Idep_range,0)

                    V_Idep_ini_hypotesis = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta,delta1).subs(t0, 0).subs(V0, -1).subs(IaA0, 0).subs(IdA0,Idep_ini_hypotesis * (Istm[i] - Ith) * (Istm[i] > Ith)).subs(Psi, psi1)
                    
                    V_Idep_ini_hypotesis_sup = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta,delta1).subs(t0,0).subs(V0, -1).subs(IaA0, 0).subs(IdA0, (Idep_ini_hypotesis_sup) * (Istm[i] - Ith) * (Istm[i] > Ith)).subs(Psi, psi1)
                    
                    V_Idep_ini_hypotesis_inf = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta,delta1).subs(t0,0).subs(V0, -1).subs(IaA0, 0).subs(IdA0, (Idep_ini_hypotesis_inf) * (Istm[i] - Ith) * (Istm[i] > Ith)).subs(Psi, psi1)
                    
                    lam_x_hyp = sym.lambdify(t, V_Idep_ini_hypotesis, modules=['numpy'])
                    lam_x_hyp_sup = sym.lambdify(t, V_Idep_ini_hypotesis_sup, modules=['numpy'])
                    lam_x_hyp_inf = sym.lambdify(t, V_Idep_ini_hypotesis_inf, modules=['numpy'])
                    #x_vals = np.linspace(0, 1000, 10001) / time_scale
                    x_vals = np.linspace(0, 400, 4001) / time_scale
                    y_vals_hyp = lam_x_hyp(x_vals)

                    y_vals_hyp_sup = lam_x_hyp_sup(x_vals)

                    y_vals_hyp_inf = lam_x_hyp_inf(x_vals)
                    sup_th_hyp = np.nonzero((y_vals_hyp > vth) * y_vals_hyp)
                    sup_th_hyp_sup = np.nonzero((y_vals_hyp_sup > vth) * y_vals_hyp_sup)
                    sup_th_hyp_inf = np.nonzero((y_vals_hyp_inf > vth) * y_vals_hyp_inf)

                    if sup_th_hyp[0].size > 0:
                        time_sp_hyp[i] = x_vals[sup_th_hyp[0][0]] * time_scale
                    else:
                        time_sp_hyp[i] = -1
                    if sup_th_hyp_sup[0].size > 0:
                        time_sp_hyp_sup[i] = x_vals[sup_th_hyp_sup[0][0]] * time_scale
                    else:
                        time_sp_hyp_sup[i] = -1
                    if sup_th_hyp_inf[0].size > 0:
                        time_sp_hyp_inf[i] = x_vals[sup_th_hyp_inf[0][0]] * time_scale
                    else:
                        time_sp_hyp_inf[i] = -1

                    err_fs_hyp[i] = abs(time_sp_hyp[i] - spk_interval[i][0])
                    err_fs_hyp_sup[i] = abs(time_sp_hyp_sup[i] - spk_interval[i][0])
                    err_fs_hyp_inf[i] = abs(time_sp_hyp_inf[i] - spk_interval[i][0])  # calcolate le differenze tra time_sp[] (che contiene gli spikes del modello) e i primi tempi per le varie correnti per il j-simo valore di Idep
            err_tot_hyp = err_fs_hyp.sum()
            err_tot_hyp_sup = err_fs_hyp_sup.sum()
            err_tot_hyp_inf = err_fs_hyp_inf.sum()

            if (err_tot_hyp <= err_tot_hyp_sup and err_tot_hyp <= err_tot_hyp_inf):  # Idep_ini_hypotesis ha errore minimo
                if (err_tot_hyp_sup - err_tot_hyp)/err_tot_hyp_sup <tol and (err_tot_hyp_inf - err_tot_hyp)/err_tot_hyp_inf <tol:  # differenza tra gli errori minore del 5% dell'errore massimo (da analizzarne il senso)
                    return Idep_ini_hypotesis, time_sp_hyp, err_tot_hyp
                else:
                    Idep_range = Idep_range / 2
                    
            elif (err_tot_hyp_sup < err_tot_hyp and err_tot_hyp_sup < err_tot_hyp_inf):  # Idep_ini_hypotesis_sup ha errore minimo
                if (err_tot_hyp - err_tot_hyp_sup)/err_tot_hyp  <tol and (err_tot_hyp_inf - err_tot_hyp_sup)/err_tot_hyp_inf  <tol:  # differenza tra gli errori minore del 5% dell'errore massimo
                    return Idep_ini_hypotesis_sup, time_sp_hyp_sup, err_tot_hyp_sup
                else:
                    Idep_ini_hypotesis = Idep_ini_hypotesis_sup
                    time_sp_hyp = time_sp_hyp_sup.copy()
                    err_tot_hyp = err_tot_hyp_sup
            
            elif (err_tot_hyp_inf < err_tot_hyp and err_tot_hyp_inf < err_tot_hyp_sup): # Idep_ini_hypotesis_inf ha errore minimo
                if  (err_tot_hyp - err_tot_hyp_inf)/err_tot_hyp  <tol and (err_tot_hyp_sup - err_tot_hyp_inf)/err_tot_hyp_sup  <tol:  # differenza tra gli errori minore del 5% dell'errore massimo
                    return Idep_ini_hypotesis_inf, time_sp_hyp_inf, err_tot_hyp_inf
                else:
                    Idep_ini_hypotesis = Idep_ini_hypotesis_inf
                    time_sp_hyp=time_sp_hyp_inf.copy()
                    err_tot_hyp=err_tot_hyp_inf
            k = k - 1

        return Idep_ini_hypotesis, time_sp_hyp, err_tot_hyp


    def loss_func(x):

        par_sc = x[0]
        tao_m = x[1]
        Ith = x[2]
        Idep_ini_vr = x[3]
        Cm = x[4]  # 189.79

        t = sym.Symbol('t')
        delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')
        [V, Iadap, Idep] = load_v3()
        min_sc = (-Cm * EL) / tao_m + (2 / (1 + vth)) * (Ith + np.sqrt(Ith * (Ith - Cm * EL * (1 + vth) / tao_m)))
        sc = min_sc + par_sc
        k_adap_min = sc * (-EL / tao_m + Ith / (Cm * (1 + vth))) / (EL ** 2)
        k_adap_max = ((Cm * EL - sc * tao_m) ** 2) / (4 * Cm * (EL ** 2) * (tao_m ** 2))
        k_adap = k_adap_min + (k_adap_max - k_adap_min) * 0.01  #
        delta1 = -Cm * EL / (sc * tao_m)
        bet = Cm * (EL ** 2) * k_adap / (sc ** 2)
        psi1 = ((-4) * bet + ((1 + delta1) ** 2)) ** (0.5)
        time_scale = 1 / (-sc / (Cm * EL))
        Vconvfact = -EL
        aux = Istm / sc
        alpha3 = aux.tolist()
        spt_00 = []
        spt_max = []
        spt_min = []
        err_fs = []
        err_ss_min = []
        err_ss_max = []

        for i in range(Istm.__len__()):

            if is_spk_2_curr[i]:
                IaA_max = Idep_ini_vr + alpha3[i] / bet + (delta1 / bet) * (1 + vrm)
                aux = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta, delta1).subs(t0, 0).subs(V0,vrm).subs(
                IaA0, IaA_max).subs(IdA0, Idep_ini_vr).subs(Psi, psi1)
                lam_x = sym.lambdify(t, aux, modules=['numpy'])
                x_vals = np.linspace(0, int(2 * max(spk_interval[i][1:])),
                                 int(2 * (max(spk_interval[i][1:]))) + 1) / time_scale
                y_vals = lam_x(x_vals)
                aus = np.nonzero((y_vals > vth) * y_vals)
                if aus[0].size > 0:
                    spt_max.append(x_vals[aus[0][0]] * time_scale)
                else:
                    spt_max.append(np.float64(10000))


                IaA_min = 0
                aux = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta, delta1).subs(t0, 0).subs(V0,vrm).subs(
                IaA0, IaA_min).subs(IdA0, Idep_ini_vr).subs(Psi, psi1)
                lam_x = sym.lambdify(t, aux, modules=['numpy'])
                x_vals = np.linspace(0, int(2 * min(spk_interval[i][1:])),
                                 int(2 * min(spk_interval[i][1:])) + 1) / time_scale
                y_vals = lam_x(x_vals)
                aus = np.nonzero((y_vals > vth) * y_vals)
                if aus[0].size > 0:
                    spt_min.append(x_vals[aus[0][0]] * time_scale)
                else:
                    spt_min.append(np.float64(10000))

                err_ss_min.append(sum(np.maximum(0, spt_min[-1] - spk_interval[i][1:])))  # max(0,(spt_min[i] - min(spk_interval[i][1:])))
                err_ss_max.append(sum(np.maximum(0, spk_interval[i][1:] - spt_max[-1])))  # max(0,max(spk_interval[i][1:]-spt_max[i]))

            #if is_spk_curr[i]:

            #    err_fs.append(abs(spt_00[i] - spk_interval[i][0]))

        [Idep_ini_alt, st0_alt, err_fsp] = calcola_Idep_2(Idep0_ini, Idep0_ini_range, 10, 0.05,Istm, sc, bet, delta1, psi1, Ith, time_scale)
        err_tot = err_fsp + sum(err_ss_min) + sum(err_ss_max)

        return err_tot

    '''
    -------------------------------------------------------------------------------
    # MAIN
    '''
    
    logfile = open(neuron_nb + "_log.txt", mode="w", encoding="utf-8")
    
     
    fitting_rule = 'monod'

    #bound = True#False
    # fitta = True
    # load_fit = False  # True
    # modalita = 'Lettura'#'Scrittura'  #





    # reads neuron experimental data
    EL, vrm, vth, Istm, spk_time_orig, dur_sign, input_start_time, spk_time, spk_interval, is_spk_curr, is_spk_2_curr = neuron_info_load(
        neuron_nb)
    


    
    
    
    try:
        # with open('neuron_' + neuron_nb + '_model_parameter.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
        #         [bet, delta1, sc,time_scale, Idep_ini_vr, tao_m,Ith,  Idep_ini,Cm]= pickle.load(f)
        with open('neuron_' + neuron_nb + '_GA_output.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                [bet, delta1, sc,time_scale, Idep_ini_vr, tao_m,Ith,Cm]= pickle.load(f)
                                        
        psi1 = ((-4) * bet + ((1 + delta1) ** 2)) ** (0.5)
        time_scale = 1 / (-sc / (Cm * EL))                 
        print('skip Genetic Algorithm')
        logfile.write('skip Genetic Algorithm\n')
    except:
        print('run Genetic Algorithm')
        logfile.write('run Genetic Algorithm\n')
        
        try:
            file1 = open('optimization_parameters.json', 'r')
            optimizer_parameters = json.load(file1)
            #n_test = optimizer_parameters["n_test_monod"]
            n_iter=optimizer_parameters["genetic_optimization_parameters"]["n_iter"]
            pop_size=optimizer_parameters["genetic_optimization_parameters"]["pop_size"]
            mutation_pr=optimizer_parameters["genetic_optimization_parameters"]["mutation_probability"]
            el_ratio=optimizer_parameters["genetic_optimization_parameters"]["elit_ratio"]
            cr_probability=optimizer_parameters["genetic_optimization_parameters"]["crossover_probability"]
            prt_portion=optimizer_parameters["genetic_optimization_parameters"]["parents_portion"]
            #print('ntest',str(n_test))
            print('optimization_parameters read')
            logfile.write('optimization_parameters read\n')
        except:
            #n_test = 100
            n_iter = 200
            pop_size = 200
            mutation_pr = 0.3
            el_ratio = 0.02
            cr_probability =0.3
            prt_portion =0.2
            print('optimization parameters from default')
            logfile.write('optimization parameters from default\n')
        
        algorithm_param = {'max_num_iteration':n_iter, \
                           'population_size': pop_size, \
                           'mutation_probability': mutation_pr, \
                           'elit_ratio':el_ratio, \
                           'crossover_probability': cr_probability, \
                           'parents_portion': prt_portion, \
                           'crossover_type': 'uniform', \
                           'max_iteration_without_improv': None}
            
        # reads parameters
        varbound = np.array([[0, 3000]] * 5)
        # cate
        if is_spk_curr[0]:#if the first curren has spikes
            varbound[2][0] = Istm[0]*0.2 #ith min
            varbound[2][1] = Istm[0]*0.9 #ith max
        else:
            #lastNotSpikingCurr = Istm[np.logical_not(is_spk_curr)][-1]
            ampiezza = Istm[is_spk_curr][0] - Istm[np.logical_not(is_spk_curr)][-1]
            varbound[2][0] = Istm[np.logical_not(is_spk_curr)][-1] + 0.1*ampiezza
            varbound[2][1] = Istm[is_spk_curr][0] - 0.1*ampiezza
        
        ################
        try:
    
            varbound[0][0] = optimizer_parameters["genetic_optimization_bounds"]["sc"]["min"]
            varbound[0][1] = optimizer_parameters["genetic_optimization_bounds"]["sc"]["max"]
    
            varbound[1][0] = optimizer_parameters["genetic_optimization_bounds"]["tao_m"]["min"]
            varbound[1][1] = optimizer_parameters["genetic_optimization_bounds"]["tao_m"]["max"]
    
            varbound[3][0] = optimizer_parameters["genetic_optimization_bounds"]["Idep_ini_vr"]["min"]
            varbound[3][1] = optimizer_parameters["genetic_optimization_bounds"]["Idep_ini_vr"]["max"]
    
            varbound[4][0] = optimizer_parameters["genetic_optimization_bounds"]["cm"]["min"]
            varbound[4][1] = optimizer_parameters["genetic_optimization_bounds"]["cm"]["max"]
  
            print('varbound from file')
            logfile.write('varbound from file\n')
    
        except:
    
            # definisco i vincoli per le variabili da ottimizzare
    
            varbound[0][1] = 100000  # par_sc
            varbound[1][0] = 10  # tao_m min
            varbound[1][1] = 100  # tao_m
            # varbound[2][0] = 200  # ith_min
            # varbound[2][1] = 299  # ith_max
            varbound[3][1] = 100  # Idep_ini_vr max
            varbound[4][1] = 100000  # cm max
 
            print('varbound from default')
            logfile.write('varbound from default\n')   
            
        # idep param used in the loss function
        try:
    
            Idep0_ini = optimizer_parameters["Idep0_search_parameters"]["Idep0_ini"]
            Idep0_ini_range = optimizer_parameters["Idep0_search_parameters"]["Idep0_ini_range"]
            n_iter_idep = optimizer_parameters["Idep0_search_parameters"]["n_iter"]
            tol = optimizer_parameters["Idep0_search_parameters"]["tolerance"]
            
            print('idep param from file')
            logfile.write('idep param from file\n')
        except:
    
            Idep0_ini = 2
            Idep0_ini_range = 2
            n_iter_idep = 200
            tol = 0.05  
            print('idep param from default')
            logfile.write('idep param from default\n')            

        model = ga(function=loss_func, dimension=5, variable_type='real', variable_boundaries=varbound,
                   algorithm_parameters=algorithm_param, convergence_curve=False,function_timeout = 100)
        print(model.param)
        logfile.write('model.param\n') 
        logfile.write(str(model.param)) 
        logfile.write('\n') 
        model.run()

        #salvo i parametri ottimizzati dall'algoritmo genetico nelle diverse variabili
        Vconvfact = -EL
        par_sc = model.best_variable[0]
        tao_m = model.best_variable[1]
        Ith = model.best_variable[2]
        Idep_ini_vr = model.best_variable[3]
        Cm = model.best_variable[4]
        
        #calcolo i parametri del modello
        min_sc = (-Cm * EL) / tao_m + (2 / (1 + vth)) * (Ith + np.sqrt(Ith * (Ith - Cm * EL * (1 + vth) / tao_m)))
        sc = min_sc + par_sc
        k_adap_min = sc * (-EL / tao_m + Ith / (Cm * (1 + vth))) / (EL ** 2)
        k_adap_max = ((Cm * EL - sc * tao_m) ** 2) / (4 * Cm * (EL ** 2) * (tao_m ** 2))
        k_adap = k_adap_min + (k_adap_max - k_adap_min) * 0.01  #
        delta1 = -Cm * EL / (sc * tao_m)
        bet = Cm * (EL ** 2) * k_adap / (sc ** 2)
        psi1 = ((-4) * bet + ((1 + delta1) ** 2)) ** (0.5)
        time_scale = 1 / (-sc / (Cm * EL))        

        with open('neuron_' + neuron_nb + '_GA_output.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([bet, delta1, sc, time_scale, Idep_ini_vr, tao_m, Ith,Cm], f)
                
                
    #carico le equazioni differenziali
    t = sym.Symbol('t')
    delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')
    [V, Iadap, Idep] = load_v3()
  
       
        # idep param used in the loss function
    try:
        file1 = open('optimization_parameters.json', 'r')
        optimizer_parameters = json.load(file1)
        Idep0_ini = optimizer_parameters["Idep0_search_parameters"]["Idep0_ini"]
        Idep0_ini_range = optimizer_parameters["Idep0_search_parameters"]["Idep0_ini_range"]
        n_iter_idep = optimizer_parameters["Idep0_search_parameters"]["n_iter"]
        tol = optimizer_parameters["Idep0_search_parameters"]["tolerance"]
        
        print('idep param from file')
        logfile.write('idep param from file\n')
    except:

        Idep0_ini = 2
        Idep0_ini_range = 2
        n_iter_idep = 200
        tol = 0.05  
        print('idep param from default')
        logfile.write('idep param from default\n')  
       
    # Idep calculation
    [Idep_ini, st0_alt, err_fsp] = calcola_Idep_2(Idep0_ini, Idep0_ini_range, n_iter_idep, tol, Istm, sc, bet, delta1,
                                                  psi1, Ith, time_scale)

    with open('neuron_' + neuron_nb + '_model_parameter.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([bet, delta1, sc, time_scale, Idep_ini_vr, tao_m, Ith, Idep_ini,Cm], f)
            
    
    aux = Istm / sc
    alpha3 = aux.tolist()
    
    first_spt_sim=calcola_primi_spike_modello(Istm,alpha3,Idep_ini,delta1,bet,psi1,time_scale,Ith,vth) #calcolo primo spike del modello per ogni corrente
    logfile.write('calcola_primi_spike_modello OK\n')
    
    time_sp,IaA,IaA_min,IaA_max,m=calcola_spikes_al_variare_Iadap(Istm, alpha3, Idep_ini_vr, delta1, bet, psi1, time_scale, vrm, vth) #restituisce i tempi dei primi spike al variare di IAdap da 0 a Adap_max
    logfile.write('calcola_spikes_al_variare_Iadap OK\n')
    
    # gets Iadap points    
    t_sp_tutti = []
    t_sp_abs_tutti = []
    Iada_tutti = []
    time_soglia = []
    for i in range(len(Istm)):
        IaA_max = Idep_ini_vr + alpha3[i] / bet + (delta1 / bet) * (1 + vrm)

        IaA = np.linspace(IaA_min, IaA_max, m)
        t_sp = []
        t_sp_abs = []
        Iada = []

        if len(spk_interval[i]) > 1:

            t_sp.append(first_spt_sim[i])
            for j in range(len(spk_interval[i]) - 1):
                if j==0:
                    t_sp.append(time_sp[i, abs(time_sp[i, :] - (spk_interval[i][j + 1]+spk_interval[i][j]-first_spt_sim[i])).argmin()])
                    Iada.append(IaA[abs(time_sp[i, :] - (spk_interval[i][j + 1]+spk_interval[i][j]-first_spt_sim[i])).argmin()])
                else:
                    t_sp.append(time_sp[i, abs(time_sp[i, :] - spk_interval[i][j + 1]).argmin()])
                    Iada.append(IaA[abs(time_sp[i, :] - spk_interval[i][j + 1]).argmin()])
                if j == 0:
                    t_sp_abs.append(t_sp[j])
                else:
                    t_sp_abs.append(t_sp_abs[j - 1] + t_sp[j])

            t_sp_tutti.append(t_sp)
            t_sp_abs_tutti.append(t_sp_abs) #tempi degli ISI per tutte le correnti
            time_soglia.append(max(t_sp_abs) + (t_sp_abs[len(t_sp_abs) - 1] - t_sp_abs[len(t_sp_abs) - 2]) / 2)
            Iada_tutti.append(Iada)


        else:
            time_soglia.append(first_spt_sim[i] / 2)
            t_sp_abs_tutti.append([])
            Iada_tutti.append([])


    with open('Iadap_time_selected_' + neuron_nb + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([t_sp_abs_tutti,Iada_tutti], f)
    
    tspk = open('Iadap_time_selected_' + neuron_nb +'.txt', mode="w", encoding="utf-8")
    for i in range(len(t_sp_abs_tutti)):
        for j in range(len(Iada_tutti[i])):
            tspk.write(str(Istm[i]) + "  " +str(t_sp_abs_tutti[i][j]) + "  " + str(Iada_tutti[i][j]) + "\n")
    tspk.close()        
    logfile.write('Iadap_time_selected WRITTEN\n')

    # monod interpolation
    try:
        n_test = optimizer_parameters["n_test_monod"]
        a = [optimizer_parameters["monod_parameters_bounds"]["a"]["min"],optimizer_parameters["monod_parameters_bounds"]["b"]["min"],optimizer_parameters["monod_parameters_bounds"]["c"]["min"],optimizer_parameters["monod_parameters_bounds"]["alpha"]["min"]]
        b = [optimizer_parameters["monod_parameters_bounds"]["a"]["max"],optimizer_parameters["monod_parameters_bounds"]["b"]["max"],optimizer_parameters["monod_parameters_bounds"]["c"]["max"],optimizer_parameters["monod_parameters_bounds"]["alpha"]["max"]]
        bound = optimizer_parameters["monod_parameters_bounds"]["bound"]
        print('monod param from file')
        logfile.write('monod param from file\n') 
        
    except:
        n_test = 100
        a = [0, -200, -200, 0]             #vincoli inferiori per i parametri della momnod
        b = [20000, 200, 200, 100000]    #vincoli superiori per i parametri della momnod
        bound=True
        print('monod param from default')
        logfile.write('monod param from default\n') 
        
    fun_loss_sel = monod_tot       #funzione di loss relativa alla monod
    
    bnds = Bounds(np.array(a), np.array(b))  #vincoli per i parametri della momnod saranno usati solo se bound==True

    best_res,best_loss=calcola_monod(n_test, fun_loss_sel, bnds, bound)

    print('**best_res: ',best_res)
    with open('best_res_' + neuron_nb + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([best_res], f)
    logfile.write('best_res.pkl WRITTEN\n')

    alpha = alpha3

    file_info = open(neuron_nb + "_info_cm.txt", mode="w", encoding="utf-8")
    file_info.write('Cm=' + str(Cm) + '\n' + 'Ith=' + str(Ith) + '\n' + 'tao=' + str(tao_m) + '\n' + 'sc=' + str(
            sc) + '\n' + 'alpha=' + str(alpha) + '\n' + 'bet=' + str(bet) + '\n' + 'delta1=' + str(
            delta1) + '\n' + 'Idep_ini=' + str(Idep_ini) + '\n' + 'Idep_ini_vr=' + str(
            Idep_ini_vr) + '\n' + 'psi=' + str(psi1) + '\n' + 'time scale=' + str(time_scale) + '\n' + 'A=' + str(
            best_res.x[0]) + '\n' + 'B=' + str(best_res.x[1]) + '\n' + 'C=' + str(
            best_res.x[2]) + '\n' + 'alpha=' + str(best_res.x[3]))
    file_info.close()
    print("Ith,tao_m,sc,alpha3,bet,delta1,Idep_ini_vr,psi1,time scale")
    print(Ith, tao_m, sc, alpha, bet, delta1, Idep_ini_vr, psi1, time_scale)
    logfile.write('neuron_info WRITTEN\n')


    logfile.write('---------------------------\n')
    logfile.write(str(optimizer_parameters))

    logfile.close()
    
    return Ith
