#!/usr/bin/env python3
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, Predictive
from jax import random
import jax
import os
import sys

import preprocessing
# from . import preprocessing
from ipynb.fs.full.models_NumPyro import model_infiniteSBP, model_finiteDPM, model_finiteDPM_extended

# post processing
import scipy
import numpy as np

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def convert_h_to_seq(h, alphabet):
    '''
    Convert numeric representation to DNA-seq representation.
    '''
    seq = [alphabet[int(h[k])] for k in range(h.shape[0])]
    return ''.join(seq)

def corrected_reads_to_fasta(posterior_samples, model, input_data, rng_key, list_read_ids, fname_output_corr, alphabet,last_x_samples=100):

    # TODO: Do we want to cut posterior samples?
    reads = input_data[1]

    # only consider the last {last_x_samples} samples to summarize results
    cluster_assignments = posterior_samples['cluster_assignments'][-last_x_samples:,:,:]
    haplotypes = posterior_samples['haplotypes'][-last_x_samples:,:,:]

    average_cluster_assignment = scipy.stats.mode(cluster_assignments, axis=0)[0][0]
    average_haplotypes = scipy.stats.mode(haplotypes, axis=0)[0][0]

    # computes empirical posterior from posterior_samples p(theta | data)
    posterior_predictive = Predictive(model, posterior_samples, infer_discrete=True)
    posterior_predictions = posterior_predictive(rng_key, input_data=input_data)['obs'][-last_x_samples:,:,:]
    posterior = [(posterior_predictions[:,n,:]==reads[n][:]).all(-1).sum() / last_x_samples for n in range(len(list_read_ids))]

    records = []
    for n ,read_id in enumerate(list_read_ids):
        hap_seq = convert_h_to_seq(average_haplotypes[average_cluster_assignment[n]][0], alphabet)
        header = '|posterior=' + str(posterior[n])
        records.append(SeqRecord(Seq(hap_seq), id=read_id, description=header))

    SeqIO.write(records, fname_output_corr, "fasta")

def haplotypes_to_fasta(posterior_samples, model, input_data, rng_key, fname_output_corr, alphabet, last_x_samples=100):
    genome_length = posterior_samples['haplotypes'].shape[2]
    alphabet_length = len(alphabet)

    # only consider the last {last_x_samples} samples to summarize results
    cluster_assignments = posterior_samples['cluster_assignments'][-last_x_samples:,:,:]
    haplotypes = posterior_samples['haplotypes'][-last_x_samples:,:,:]

    # number of reads assigned to each haplo
    hap_ids, ave_reads = np.unique(cluster_assignments, return_counts=True)

    # collapse haplotypes into unique set
    average_haplotypes = scipy.stats.mode(haplotypes, axis=0)[0][0]
    unique_haplotypes = np.unique(average_haplotypes, axis=0)
    idx_collapsed_haplotypes = [np.where(np.all(average_haplotypes==unique_hap,axis=1)) for unique_hap in unique_haplotypes]
    idx_collapsed_haplotypes = [np.intersect1d(idx_col,hap_ids) for idx_col in idx_collapsed_haplotypes]
    # map idx_collapsed_haplotypes to hap_ids order
    outer = []
    for idx_collapsed in idx_collapsed_haplotypes:
        outer.append([np.argwhere(hap_ids==i)[0][0] for i in idx_collapsed])
    mapped_idx_collapsed_haplotypes = outer

    # collapse number of average reads to unique haplotypes
    ave_reads = np.array([np.sum(ave_reads[idx_hap]) for idx_hap in mapped_idx_collapsed_haplotypes])
    ave_reads = ave_reads / last_x_samples

    # empirical posteriors of haplotypes (collapsed like in ave_reads computation)
    posterior_predictive = Predictive(model,
                                      posterior_samples,
                                      infer_discrete=True,
                                      return_sites=['haplotypes']
                                     )
    posterior_predictions = posterior_predictive(rng_key,
                                                 input_data=input_data
                                                )['haplotypes'][-last_x_samples:,:,:]
    posterior = [(posterior_predictions[:,hap_ids[idx_hap],:]==average_haplotypes[hap_ids[idx_hap]][:]).all(-1).sum() / last_x_samples for idx_hap in mapped_idx_collapsed_haplotypes]

    # write to fasta
    records = []
    for k in range(unique_haplotypes.shape[0]):
        head = ' | posterior='+str(posterior[k])+' ave_reads='+str(ave_reads[k])
        seq = convert_h_to_seq(unique_haplotypes[k], alphabet)
        records.append(SeqRecord(Seq(seq), id='haplotype'+str(k), description=head))
    SeqIO.write(records, fname_output_corr, "fasta")


def main(freads_in, fref_in, output_dir, cluster_num, str_model, alphabet = 'ACGT-'):

    #window_id = freads_in.split('/')[-1][:-4] # freads_in is absolute path
    #window=[int(window_id.split('-')[2])-1,int(window_id.split('-')[3].split('.')[0])]
    #output_name = output_dir+window_id+'-'

    if str_model == "infiniteSBP":
        model = model_infiniteSBP
    elif str_model == "finiteDPM":
        model = model_finiteDPM
    elif str_model == "finiteDPM_extended":
        model = model_finiteDPM_extended
    else:
        raise ValueError("Input model is not defined.")

    if os.path.exists(output_dir)==False: # Check whether the specified path exists or not
        os.makedirs(output_dir)

    alphabet_length = len(alphabet) # size alphabet

    reference = preprocessing.fasta2ref(fref_in, alphabet)
    reads, list_read_ids = preprocessing.fasta2reads(freads_in, alphabet)

    input_data = reference, reads, alphabet_length


    rng_key = jax.random.PRNGKey(0)
    num_samples = 100000
    num_warmup = int(num_samples / 2)

    # Run NUTS
    kernel = NUTS(model)
    mcmc = MCMC(
        DiscreteHMCGibbs(kernel),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
         progress_bar=False
    )
    mcmc.run(rng_key, input_data)

    #mcmc.print_summary()

    posterior_samples = mcmc.get_samples()

    haplotypes_to_fasta(posterior_samples, model, input_data, rng_key, output_dir+'support.fas' , alphabet, last_x_samples=100)
    corrected_reads_to_fasta(posterior_samples, model, input_data, rng_key, list_read_ids, output_dir+'cor.fas', alphabet,last_x_samples=100)

    """
    Check convergence of the chain every x samples.
    1. Check convergence by looking at Gelman Rubin and ESS
    Gelman Rubin: https://arxiv.org/pdf/1812.09384.pdf
    \hat{R} <1.1 or 1.01
    2. Continue running the chain using  post_warmup_state
    Example:
    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    mcmc.run(random.PRNGKey(0))
    first_100_samples = mcmc.get_samples()
    mcmc.post_warmup_state = mcmc.last_state
    mcmc.run(mcmc.post_warmup_state.rng_key)  # or mcmc.run(random.PRNGKey(1))
    second_100_samples = mcmc.get_samples()
    """


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
#(freads_in, fref_in, output_dir, num_clusters, model, alphabet = 'ACGT-')
# ./run_mcmc.py ../../../test_data/super_small_ex2/seqs.fasta ../../../test_data/super_small_ex2/ref.fasta ./Output/ 10 finiteDPM ACGT-
