import numpy as np
from timeit import default_timer as timer
import pickle
import pandas as pd
import multiprocessing as mp
from scipy.special import digamma
from scipy.stats._multivariate import _lnB as lnB
from scipy.special import betaln


# my python scripts
from . import initialization
from . import update_eqs
from . import elbo_eqs
from . import analyze_results


"""
Parallelizing with Pool following:
https://www.machinelearningplus.com/python/parallel-processing-python/
"""

results =[]

def collect_result(result):
    global results
    results.append(result)


def multistart_cavi(K, alpha0, alphabet, reference_binary, reference_seq, reads_list,reads_seq_binary, reads_weights, n_starts, output_dir):

    results =[]

    for start in range(n_starts):
        res_run=run_cavi(K, alpha0, alphabet, reference_binary, reference_seq, reads_list, reads_seq_binary, reads_weights,start, output_dir)
        results.append(res_run)


    """
    Shorah is already using pool for the windows it is not possible to h
    pool = mp.Pool(mp.cpu_count())

    for start in range(n_starts):

        pool.apply_async(run_cavi, args=(K, alpha0, alphabet, reference_binary, reference_seq, reads_list, reads_seq_binary, reads_weights,start, output_dir), callback=collect_result)

    pool.close()
    pool.join()
    """

    return results


def run_cavi(K, alpha0, alphabet, reference_binary, reference_seq, reads_list, reads_seq_binary, reads_weights,start_id, output_dir):
    """
    Runs cavi (coordinate ascent variational inference).
    """
    start_time =timer()
    dict_result={'run_id': start_id,
                 'N': len(reads_list),
                 'K': K,
                 'alpha0': alpha0,
                 'alphabet': alphabet} # N= #reads, K= #components


    state_init_dict = initialization.draw_init_state(K, alpha0, alphabet, reads_list, reference_binary)

    state_init_dict.update({'lnB_alpha0': lnB(state_init_dict['alpha']),
                            'betaln_a0_b0': betaln(state_init_dict['gamma_a'],state_init_dict['gamma_b']),
                            'betaln_c0_d0': betaln(state_init_dict['theta_c'],state_init_dict['theta_d'])})

    # write initial values to dict
    dict_result.update({'theta0': state_init_dict['mean_log_theta'][0],
                        'gamma0': state_init_dict['mean_log_gamma'][0],
                        'mean_h0': state_init_dict['mean_haplo'],
                        'mean_z0': state_init_dict['mean_cluster'],
                        'mean_log_pi': state_init_dict['mean_log_pi']})

    list_history = [state_init_dict]
    end_time_intit =timer()
    dict_result.update({'time_initialization': end_time_intit-start_time})

    # Iteratively update mean values
    iter=0
    message='' # those can be deleted afterwardsd
    exitflag=''# those can be deleted afterwardsd
    converged=False
    elbo=0
    elbo_history=[]
    state_curr_dict = state_init_dict
    k=0

    while (converged==False) or (k<15):

        if iter<=1:
            digamma_alpha_sum=digamma(state_curr_dict['alpha'].sum(axis=0))
            digamma_a_b_sum=digamma(state_curr_dict['gamma_a']+state_curr_dict['gamma_b'])
            digamma_c_d_sum=digamma(state_curr_dict['theta_c']+state_curr_dict['theta_d'])
            state_curr_dict.update({'digamma_alpha_sum': digamma_alpha_sum})
            state_curr_dict.update({'digamma_a_b_sum': digamma_a_b_sum})
            state_curr_dict.update({'digamma_c_d_sum': digamma_c_d_sum})

        state_curr_dict = update_eqs.update(reads_seq_binary, reads_weights,reads_list, reference_binary, state_init_dict, state_curr_dict)
        elbo_old = elbo
        #print('ELBO ',elbo)
        elbo = elbo_eqs.compute_elbo(reads_weights,reads_seq_binary,reference_binary, state_init_dict, state_curr_dict)

        elbo_history.append(elbo)

        if iter>1:
            if (elbo_history[-2]>elbo) and np.abs(elbo-elbo_history[-2])>1e-08 :
                message='Error: ELBO is decreasing.'
                exitflag=-1
                break
            elif np.abs(elbo-elbo_history[-2]) <1e-08:
                converged=True
                k+=1
                message='ELBO converged.'
                exitflag=0
            else:
                k=0

        #if k%10==0: # every 10th parameter set is saved to history
        state_curr_dict.update({'elbo':elbo})
        list_history.append(state_curr_dict)

        iter+=1
    # End: While-loop

    state_curr_dict.update({'elbo':elbo})
    # save history
    list_history.append(state_curr_dict)
    df_history = pd.DataFrame(list_history)
    df_history.to_csv(output_dir+'history_run'+str(start_id)+'.csv')

    end_time_optimization = timer()

    dict_result.update({'exit_message': message,
                        'exitflag': exitflag,
                        'n_iterations': iter,
                        'converged': converged,
                        'elbo': elbo,
                        'elbo_history': elbo_history,
                        'time_optimization': end_time_optimization-end_time_intit,
                        'runtime_total': end_time_optimization-start_time})

    dict_result.update(state_curr_dict)
    summary = analyze_results.summarize_results(state_curr_dict,
                                                         alphabet,
                                                         reads_seq_binary,
                                                         reads_weights,
                                                         reads_list,
                                                         reference_binary,
                                                         reference_seq)
    dict_result.update(summary)
    state_curr_dict.update(summary)


    f2 = open(output_dir+'results'+str(start_id)+'.pkl',"wb")
    pickle.dump(dict_result,f2)
    f2.close()

    result=(state_init_dict, state_curr_dict)

    return result
