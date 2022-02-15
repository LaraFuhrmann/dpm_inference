#!/usr/bin/env python3
import sys
import pandas as pd
import skbio
from skbio import Sequence
from skbio.sequence.distance import hamming
import random
import numpy as np
from timeit import default_timer as timer
import os
import pickle
import multiprocessing as mp

import auxiliary_dpm_sampling as aux_dpm
import convergence_diagnostics as conv_diag
import extra
import preparation

def run_chain(conv_lst, chain_id,reads_list,reference_seq,alpha,max_iter,alphabet,window_size,totbases,total_bases_counts,output_dir,dict_result,results_list):

    time_start_chain = timer()
    dict_runtime={'chain_id': chain_id}
    dict_result.update({'chain_id': chain_id})

    outfile_chain = open(output_dir+'output_info_chain'+str(chain_id)+'.txt','w')
    outfile_chain.write('Number of reads '+ str(len(reads_list))+'\n')

    # initializing the clusters - each read in one cluster
    list_of_clusters = preparation.init_clusters(reads_list,reference_seq)
    theta = 0.7 +(0.95-0.7)*random.uniform(0, 1) # 1- theta is the error rate
    gamma = 0.7 +(0.95-0.7)*random.uniform(0, 1) # 1- gamma is the mutation rate
    state = aux_dpm.State(cluster_list=list_of_clusters, theta=theta, gamma=gamma,window_size=window_size, alphabet=alphabet, alpha=alpha)
    for temp_cluster in list_of_clusters:
        temp_cluster.haplotype_seq = aux_dpm.sample_haplotype(temp_cluster.reads_idx_list, reads_list, state, total_bases_counts, totbases)
        temp_cluster.update_distances(reads_list,reference_seq)

    #df_history = preparation.history_dataframe(n_reads=len(reads_list))
    #cols_his =  ['n_iter', 'alpha', 'alphabet', 'n_cluster', 'haplotypes']
    cols_his =  ['n_iter', 'alpha', 'alphabet', 'n_cluster']
    cols_his += ['c_'+str(n) for n in range(len(reads_list))]
    cols_his += ['gamma', 'theta']
    cols_his += ['ess_max','gelman_rubin']
    cols_his += ['conv_ess_max', 'conv_geweke','conv_gelman_rubin']
    df_history = pd.DataFrame(columns=cols_his)
    # TODO: write inital point to history

    sample_points_time_series =[]
    gewecke_statisitc =[]
    ess_convergence=False
    conv_ess_max = False
    conv_geweke = False
    conv_gelman_rubin = False
    conv_dia = [conv_ess_max, conv_geweke, conv_gelman_rubin]
    chain_converged=False
    k=0

    while (k<max_iter) and (not chain_converged):
        k+=1

        #go through unique reads
        for idx_read, read in enumerate(reads_list):
            # sample cluster for read
            aux_dpm.sample_class(idx_read, reads_list,reference_seq, state,total_bases_counts, totbases)

        n_cluster=len(list_of_clusters)
        dt=0.0 # matches over all reads
        dk1=0
        hapbases=0

        for curr_cluster in list_of_clusters: #iterate over all clusters

            curr_cluster.haplotype_seq=aux_dpm.sample_haplotype(curr_cluster.reads_idx_list, reads_list,state, total_bases_counts, totbases)
            curr_cluster.update_distances(reads_list,reference_seq) # update distances to newly sampled haplotype

            for i in curr_cluster.reads_idx_list:
                dt+=curr_cluster.matches2reads[i]*reads_list[i].metadata['weight']

            dk1+=curr_cluster.matches2reference
            hapbases+=curr_cluster.matches2reference+curr_cluster.distance2reference

        eps1 =  0.985
        eps2 = 0.001
        b_alpha=dt+(eps1*eps2*totbases)
        b_beta=(totbases - dt) + eps2 * totbases * (1 - eps1)
        theta = np.random.beta(a=b_alpha, b= b_beta,size=1)[0]
        theta=np.fmax(np.fmin(theta, np.nextafter(1, 0)),1-np.nextafter(1, 0))
        gamma= dk1/hapbases

        state.theta = theta
        state.gamma = gamma
        state.cluster_list= list_of_clusters

        dict_history = state.to_dict(reads_list)

        # check convergence
        if k >50 and k%50==0:

            conv_ess_max, ess_max_chain =conv_diag.check_convergence_ess_max(df_history, 0.5)
            dict_history.update({'conv_ess_max': ess_max_chain,
                               'conv_ess_max':conv_ess_max})
            #print('conv_ess_max ', conv_ess_max)
            conv_geweke = conv_diag.check_convergence_geweke(df_history, k)
            dict_history.update({'conv_geweke':conv_geweke})
            #print('conv_geweke ', conv_geweke)

            # check what other chain is doing
            for item in conv_lst:
                if not isinstance(item, pd.DataFrame):
                    if item == "converged":
                        print("gelman_rubin converged")
                        conv_gelman_rubin=True
            if not conv_gelman_rubin:
                conv_lst[chain_id]=df_history

                list_df_chains = [df_t for df_t in conv_lst if isinstance(df_t, pd.DataFrame)]
                if len(list_df_chains)>1: # a minimum of 2 chains is needed for Gelman-Rubin
                    conv_gelman_rubin, rhat = conv_diag.check_convergence_gelman_rubin(list_df_chains, threshold_rhat=1.01)
                    dict_history.update({'conv_gelman_rubin':conv_gelman_rubin,
                                         'gelman_rubin': rhat.eval})

                    if conv_gelman_rubin:
                        conv_lst[chain_id]=True

            dict_history.update({'conv_gelman_rubin':conv_gelman_rubin})
            conv_dia = [conv_ess_max, conv_geweke, conv_gelman_rubin]

            values, counts = np.unique(conv_dia, return_counts=True)

            if 'True' in values:
                idx_temp = np.where(values=='True')
                if counts[idx_temp]>=2:
                    chain_converged=True

        dict_history.update({'n_iter': k})
        #df_history = df_history.append(dict_history, ignore_index=True)
        df_new_row = pd.DataFrame(dict_history, index=[0])
        df_history = pd.concat([df_history, df_new_row])

    df_history.to_csv(output_dir+'history_chain'+str(chain_id)+'.csv')
    end_time_total = timer()
    dict_runtime.update({'n_iter': k})
    #dict_runtime.update({'avg_time_per_sampling_iter': (end_time_total-time_start_chain)/k})
    dict_runtime.update({'time_total': end_time_total-time_start_chain})
    f = open(output_dir+'runtime_chain'+str(chain_id)+'.pkl',"wb")
    pickle.dump(dict_runtime,f)
    f.close()

    dict_result.update({'n_reads': len(reads_list),
                   'alpha': alpha,
                   'alphabet': alphabet,
                   'max_iter': max_iter,
                   'n_iter': k,
                   'time_total': end_time_total-time_start_chain,
                   'reads_list': reads_list,
                   'list_of_clusters': list_of_clusters})

    for i, temp_cluster in enumerate(list_of_clusters):
        assigned_reads=[]
        for idx in temp_cluster.reads_idx_list:
            assigned_reads+=reads_list[idx].metadata['identical_reads']
        dict_result.update({'haplotype'+str(i): temp_cluster.haplotype_seq,
                            'weight'+str(i): aux_dpm.weight_of_cluster(temp_cluster,reads_list),
                            'assignedReads'+str(i): assigned_reads
                            })
    dict_result.update({'conv_ess_max':conv_ess_max, 'conv_geweke': conv_geweke, 'conv_gelman_rubin': conv_gelman_rubin})

    results_list.append(dict_result)

    outfile_chain.write('------------------'+'\n')
    outfile_chain.write('End of sampling'+'\n')
    outfile_chain.write('------------------'+'\n')

    outfile_chain.write('Cluster information'+'\n')
    for i, temp_cluster in enumerate(list_of_clusters):
        outfile_chain.write('cluster '+ str(i)+': '+'\n')
        outfile_chain.write('reads cluster index: ' +str([reads_list[idx].metadata['cluster'] for idx in temp_cluster.reads_idx_list])+'\n')
        outfile_chain.write('reads: ' +str(assigned_reads)+'\n')
        outfile_chain.write('cluster weight: ' +str(aux_dpm.weight_of_cluster(temp_cluster,reads_list))+'\n')
        outfile_chain.write('haplotype sequence: '+str(temp_cluster.haplotype_seq)+'\n')
        outfile_chain.write('distance to assigned reads: '+ str([ temp_cluster.distance2reads[j] for j in temp_cluster.reads_idx_list])+'\n')
        outfile_chain.write('distance to reference: '+ str(temp_cluster.distance2reference)+'\n')
        outfile_chain.write('assigned reads distance to reference: '+ str([ reads_list[j].metadata['distance2reference'] for j in temp_cluster.reads_idx_list])+'\n')
        outfile_chain.write('Compare haplotype to its assigned reads\n')
        for read_i in temp_cluster.reads_idx_list:
            outfile_chain = extra.sequence_compare(temp_cluster.haplotype_seq, reads_list[read_i], outfile_chain)
        outfile_chain.write('Compare haplotype to reference\n')
        outfile_chain = extra.sequence_compare(temp_cluster.haplotype_seq, reference_seq, outfile_chain)

    outfile_chain.close()

    #return dict_result


def main(freads_in, fref_in, output_dir, alpha= 0.01, max_iter=1000):
    # max_iter in the orignal Shorah is 1000 (at least in dom_sampler.hpp)
    start_time = timer()
    dict_runtime={}

    if os.path.exists(output_dir)==False: # Check whether the specified path exists or not
        os.makedirs(output_dir) # Create a new directory because it does not exist

    reference_seq = preparation.load_reference_seq(fref_in)
    reads_list = preparation.load_fasta2reads_list(freads_in)

    for read in reads_list: read.metadata.update({'distance2reference': aux_dpm.distance(read, reference_seq)})
    for read in reads_list: read.metadata.update({'matches2reference': aux_dpm.matches(read, reference_seq)})

    # global variables
    alphabet = ['A','C','G','T','-']
    window_size=len(reference_seq.values)
    B = len(alphabet) # size of alphabet {A,C,T,G,-}

    totbases = preparation.count_totbases(reads_list)
    total_bases_counts = preparation.base_counts_per_position(reads_list, window_size, alphabet)

    ### Run multiple chains to compare for convergence detection
    m = mp.Manager()

    conv_lst = m.list()
    results_list= m.list()

    n_chains = 2
    conv_lst.append(None)
    conv_lst.append(None)

    dict_result={'fname_reads_in': freads_in,
                 'fname_ref_in': fref_in}

    process_chain1 = mp.Process(target=run_chain,args=(conv_lst, 0,reads_list,reference_seq,alpha,max_iter,alphabet,window_size,totbases,total_bases_counts,output_dir, dict_result,results_list, ))
    process_chain2 = mp.Process(target=run_chain,args=(conv_lst, 1,reads_list,reference_seq,alpha,max_iter,alphabet,window_size,totbases,total_bases_counts,output_dir, dict_result,results_list, ))

    process_chain1.start()
    process_chain2.start()

    process_chain2.join()
    process_chain1.join()

    new_list=[]

    for result_dict_temp in results_list:
        new_result={'fname_reads_in': freads_in,
                    'fname_ref_in': fref_in,
                    'n_reads': result_dict_temp['n_reads'],
                    'alpha': result_dict_temp['alpha'],
                    'alphabet': result_dict_temp['alphabet'],
                    'max_iter': result_dict_temp['max_iter'],
                    'n_iter': result_dict_temp['n_iter'],
                    'time_total': result_dict_temp['time_total'],
                    'reads_list': result_dict_temp['reads_list'],
                    #'list_of_clusters': result_dict_temp['list_of_clusters'],
                    'chain_id': result_dict_temp['chain_id'],
                    'conv_ess_max':result_dict_temp['conv_ess_max'],
                    'conv_geweke': result_dict_temp['conv_geweke'],
                    'conv_gelman_rubin': result_dict_temp['conv_gelman_rubin']
                    }

        for i, temp_cluster in enumerate(result_dict_temp['list_of_clusters']):
                assigned_reads=[]
                for idx in temp_cluster.reads_idx_list:
                    assigned_reads+=reads_list[idx].metadata['identical_reads']
                new_result.update({'haplotype'+str(i): temp_cluster.haplotype_seq,
                                    'weight'+str(i): aux_dpm.weight_of_cluster(temp_cluster,reads_list),
                                    'assignedReads'+str(i): assigned_reads
                                    })
        new_list.append(new_result)

    f3 = open(output_dir+'result.pkl',"wb")
    pickle.dump(new_list,f3)
    f3.close()

    end_time_total = timer()
    dict_runtime.update({'time_total': end_time_total-start_time})

    f = open(output_dir+'runtime_total.pkl',"wb")
    pickle.dump(dict_runtime,f)
    f.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
