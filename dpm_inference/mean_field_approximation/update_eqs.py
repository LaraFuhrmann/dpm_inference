from scipy.special import digamma
from scipy.special import betaln
import numpy as np
#from timeit import default_timer as timer


def update(reads_seq_binary, reads_weights,reads_list, reference_binary, state_init, state_curr):

    #sssstart_time = timer()
    alpha0 = state_init['alpha']
    a = state_init['gamma_a']
    b = state_init['gamma_b']
    c = state_init['theta_c']
    d = state_init['theta_d']

    mean_z = state_curr['mean_cluster']
    mean_h = state_curr['mean_haplo']

    #alpha_updated = state_curr['alpha']
    mean_log_pi = state_curr['mean_log_pi']
    digamma_alpha_sum = state_curr['digamma_alpha_sum']

    #a_updated = state_curr['gamma_a']
    #b_updated = state_curr['gamma_b']
    mean_log_gamma = state_curr['mean_log_gamma']
    digamma_a_b_sum=state_curr['digamma_a_b_sum']
    digamma_c_d_sum=state_curr['digamma_c_d_sum']

    #c_updated = state_curr['theta_c']
    #d_updated = state_curr['theta_d']
    mean_log_theta = state_curr['mean_log_theta']
    #start_time = timer()
    #old = mean_h
    mean_h = update_mean_haplo(reads_seq_binary, reads_weights, reads_list,reference_binary,mean_z,mean_log_theta,mean_log_gamma)
    #print('diff', old-mean_h)
    #print('time update_mean_haplo ', timer()-start_time)
    #start_time = timer()
    mean_z = update_mean_cluster(reads_seq_binary, reads_list,mean_log_pi,mean_h,mean_log_theta)
    #mean_h = update_eqs.update_mean_haplo(reads_list,reference_binary,mean_z,mean_log_theta,mean_log_gamma)
    #print('time update_mean_cluster ', timer()-start_time)
    #start_time = timer()
    alpha_updated = update_alpha(alpha0, mean_z, reads_list,reads_weights)
    #print('time update_alpha ', timer()-start_time)
    #start_time = timer()
    mean_log_pi = get_mean_log_pi(alpha_updated, digamma_alpha_sum)
    #print('time get_mean_log_pi ', timer()-start_time)
    #start_time = timer()

    a_updated,b_updated = update_a_and_b(reference_binary,mean_h,a,b)

    mean_log_gamma =get_mean_log_beta_dist(a_updated,b_updated,digamma_a_b_sum)
    #print('time mean_log_gamma ', timer()-start_time)
    #start_time = timer()
    c_updated,d_updated = update_c_and_d(reads_seq_binary, reads_weights,reads_list, mean_z, mean_h, c,d)
    mean_log_theta = get_mean_log_beta_dist(c_updated,d_updated,digamma_c_d_sum)
    #print('time get_mean_thetha ', timer()-start_time)
    state_curr_dict_new = dict({'alpha': alpha_updated,
                            'mean_log_pi': mean_log_pi,
                            'theta_c': c_updated,
                            'theta_d': d_updated,
                            'mean_log_theta': mean_log_theta,
                            'gamma_a': a_updated,
                            'gamma_b': b_updated,
                            'digamma_alpha_sum':digamma_alpha_sum,
                            'digamma_c_d_sum':digamma_c_d_sum,
                            'digamma_a_b_sum':digamma_a_b_sum,
                            'mean_log_gamma': mean_log_gamma,
                            'mean_haplo': mean_h,
                            'mean_cluster': mean_z
                            })
    #print('one update ', timer()-sssstart_time)


    return state_curr_dict_new

def get_mean_log_pi(alpha, digamma_alpha_sum):
    """
    Note that the digamma function can be inefficient.
    """
    #digamma_alpha_sum = digamma(alpha.sum(axis=0))
    mean_log_pi = digamma(alpha)-digamma_alpha_sum
    return mean_log_pi

def get_mean_log_beta_dist(a,b,digamma_sum):
    # I tested this one and it seemed not to make the difference
    #digamma_sum = digamma(a+b)
    mean_log_gamma = digamma(a)-digamma_sum
    mean_log_gamma_inv = digamma(b)-digamma_sum
    return mean_log_gamma, mean_log_gamma_inv

def update_mean_cluster(reads_seq_binary,reads_list,mean_log_pi,mean_haplo,mean_log_theta):
    #L = mean_haplo.shape[1] # length of seq
    #K = mean_haplo.shape[0]
    #N = reads_seq_binary.shape[0]

    B = mean_haplo.shape[2]

    temp_haplo_k = mean_log_theta[0]*mean_haplo
    temp_haplo_k_inv = (mean_log_theta[1]-np.log(B-1))*mean_haplo

    temp_c = np.einsum('NLB,KLB->NK',reads_seq_binary,temp_haplo_k)
    temp_c += np.einsum('NLB,KLB->NK',(1-reads_seq_binary),temp_haplo_k_inv)
    temp_c[:] += mean_log_pi

    #print('temp_c ', temp_c)

    max_z = np.max(temp_c, axis=1)
    max_z = max_z[:, np.newaxis]
    mean_z= np.exp(temp_c-max_z)
    c_normalize = mean_z.sum(axis=1)
    c_normalize = c_normalize[:, np.newaxis]
    mean_z = mean_z/c_normalize

    return mean_z

def update_mean_haplo(reads_seq_binary, reads_weights, reads_list,reference_table, mean_cluster,mean_log_theta,mean_log_gamma):
    B=reference_table.shape[1] # size of alphabet

    ref_part=reference_table*mean_log_gamma[0]+(1-reference_table)*(mean_log_gamma[1]-np.log(B-1))
    b1=mean_log_theta[0]
    b2=(mean_log_theta[1]-np.log(B-1))

    all_N_pos=reads_seq_binary.sum(axis=2)>0# if reads_list[n].seq_binary[l].sum(axis=0)=0 then "N" at position l then position l is ignored
    temp_sum = np.add(np.ones(reads_seq_binary.shape),(-1)*reads_seq_binary)
    reads_seq_binary_inv = np.einsum('NL,NLB->NLB',all_N_pos, temp_sum)

    mean_cluster_weight = np.einsum('N,NK->NK',reads_weights,mean_cluster)

    log_mean_haplo = b1*np.einsum('NLB,NK->KLB', reads_seq_binary, mean_cluster_weight) # shape: (K,L,B)
    log_mean_haplo += b2*np.einsum('NLB,NK->KLB', reads_seq_binary_inv, mean_cluster_weight)
    log_mean_haplo[:] += ref_part

    max_hap = np.max(log_mean_haplo, axis=2) # shape: (K,L)
    max_hap = max_hap[:, :, np.newaxis]
    mean_haplo = np.exp(log_mean_haplo-max_hap)
    c_normalize=mean_haplo.sum(axis=2)
    c_normalize = c_normalize[:,:, np.newaxis]
    mean_haplo= mean_haplo/c_normalize

    return mean_haplo

def update_a_and_b(reference_table,mean_haplo,a,b):
    # update a and b for mutation rate gamma

    up_a = a.copy()
    up_a += np.einsum('KLB,LB->',mean_haplo,reference_table)

    up_b = b.copy()
    up_b += np.einsum('KLB,LB->',mean_haplo,(1-reference_table))

    return up_a,up_b

def update_c_and_d(reads_seq_binary, reads_weights,reads_list, mean_cluster, mean_haplo, c,d):

    mean_cluster_weight = np.einsum('N,NK->NK',reads_weights,mean_cluster)

    up_c = c.copy()
    temp_c = np.einsum('NLB,KLB->NK',reads_seq_binary,mean_haplo)
    up_c += np.einsum('NK,NK->',temp_c,mean_cluster_weight)

    up_d = d.copy()
    temp_c = np.einsum('NLB,KLB->NK',reads_seq_binary,(1-mean_haplo))
    up_d += np.einsum('NK,NK->',temp_c,mean_cluster_weight)

    return up_c,up_d

def update_alpha(alpha, mean_cluster, reads_list,reads_weights):

    temp_alpha = alpha.copy()
    temp_mean_cluster = mean_cluster.copy()
    mean_cluster_weight = np.einsum('N,NK->NK',reads_weights,temp_mean_cluster)
    temp_alpha+=mean_cluster_weight.sum(axis=0)

    return temp_alpha
