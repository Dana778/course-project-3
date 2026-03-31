import json
import multiprocessing
import sys
from pprint import pprint

import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.stats import poisson

import obs  # module for creating observation sequence


def create_observations(tsv, bed):
    """Generates observation sequences from input data and determines state dimensions."""
    try:
        result = obs.process_data(tsv, bed)
        print(' Observation sequences for HMM created successfully!')
    except Exception as e:
        print(f"!!! Critical error: {e}")
        sys.exit(1)
    
    # Get number of states
    max_val, max_info = obs.get_number_states(result)
    print(f'Max differences in 1000bp window: {max_val + 1}')
    
    return result, max_val + 1


def prepare_matrices_from_dict(data_dict):
    """Converts haplotype dictionary into NumPy matrices (O1, O2) preserving insertion order."""
    
    # 1. Keep original insertion order
    hap_names = list(data_dict.keys())
    
    M = len(hap_names)  # Number of haplotypes
    if M == 0:
        raise ValueError("Dictionary contains no haplotypes.")
    
    N = len(data_dict[hap_names[0]])  # Number of windows
    
    # 2. Initialize matrices
    O1 = np.zeros((M, N), dtype=np.int32)
    O2 = np.zeros((M, N), dtype=np.int32)
    
    # 3. Fill matrices
    for i, hap in enumerate(hap_names):
        hap_data = np.array(data_dict[hap])  # shape (N, 2)
        O1[i, :] = hap_data[:, 0]  # Column 0 → O1
        O2[i, :] = hap_data[:, 1]  # Column 1 → O2
        
    return O1, O2, hap_names


# Ti: Introgression of Nd
# Taf: Time out of Africa
# Tn: Time of Split between Nd and Sapiens

# Transition probabilities
def initA(L, rr, Ti, a):
    """Calculates standard transition probability matrix based on recombination and admixture."""
    A = np.zeros((2, 2))
    
    A[0][1] = Ti * rr * L * a
    A[0][0] = 1 - A[0][1]
    
    A[1][0] = Ti * rr * L * (1 - a)
    A[1][1] = 1 - A[1][0]
    
    return A

# Log-transition probabilities
def get_log_A(L, rr, Ti, a):
    """Calculates transition probabilities in log-space to prevent numerical underflow."""
    A = np.zeros((2, 2))
    
    # Probability of recombination event
    prob = Ti * rr * L
    if prob > 0.5: 
        prob = 0.5  # Safety cap
    
    # Transitions: State 0 → 1
    A[0, 1] = prob * a
    A[0, 0] = 1.0 - A[0, 1]
    
    # Transitions: State 1 → 0
    A[1, 0] = prob * (1.0 - a)
    A[1, 1] = 1.0 - A[1, 0]
    
    return np.log(A + 1e-300)

# a_old - доля примеси сразу после старой волны
# a_young - доля примеси сразу после новой волны 
def initA3(L, rr, t_old, t_young, a_old, a_young):
    """Calculates standard transition probability matrix based on recombination and admixture."""
    A = np.zeros((3, 3), dtype=float)
    
    A[0][1] = (1-np.exp(-rr*L*(t_old-t_young)))*np.exp(-rr*L*t_young)*a_old + (1-np.exp(-rr*L*t_young))*a_old*(1-a_young)  # Modern -> Archaic1
        # 1-np.exp(-rr*L*(t_old-t_young)) - между старой и новой волной произошла рекомбинация
        # np.exp(-rr*L*t_young) - не было рекомбинации после 2 волны
        # 1-a_young - доля генома, которая НЕ стала archaic2 сразу после 2-й волны 
    A[0][2] = (1-np.exp(-rr*L*t_young))*a_young  # Modern -> Archaic2
    A[0][0] = 1 - A[0][1] - A[0][2]  # Modern -> Modern, рекомбинации вообще не было, либо рекомбинация была, но новый кусок снова оказался modern
    
    A[1][0] = (1-np.exp(-rr*L*(t_old-t_young)))*np.exp(-rr*L*t_young)*(1-a_old) + (1-np.exp(-rr*L*t_young))*(1 - a_old)*(1-a_young) # Archaic1 -> Modern
        # (1-np.exp(-rr*L*(t_old-t_young)))*np.exp(-rr*L*t_young) - рекомбинация произошла после 1 волны, после 2 волны рекомбинации не было (то есть примешалось archaic1)
        # 1-a_old - попали в ту часть генома, которая от африканцев (modern)
        # 1-np.exp(-rr*L*t_young) - произошла рекомбинация после 2 волны
        # (1 - a_old)*(1-a_young) - попали в ту часть генома, которая от африканцев (modern)
    A[1][2] = (1-np.exp(-rr*L*t_young))*a_young  # Archaic1 -> Archaic2 
    A[1][1] = 1 - A[1][0] - A[1][2]

    A[2][0] = (1-np.exp(-rr*L*t_young))*(1 - a_old)*(1-a_young) # Archaic2 -> Modern
    A[2][1] = (1-np.exp(-rr*L*t_young))*a_old*(1-a_young)  # Archaic2 -> Archaic1 
    A[2][2] = 1 - A[2][0] - A[2][1] 
    
    A = np.clip(A, 0.0, 1.0)
    A = A / A.sum(axis=1, keepdims=True) # нормировка строк: делим каждую строку на её сумму, чтобы в каждой строке сумма стала 1
    return A

def get_log_A3(L, rr, transition_params):
    t_old, t_young, a_old, a_young = transition_params[0], transition_params[1], transition_params[2], transition_params[3]
    A = initA3(L, rr, t_old, t_young, a_old, a_young)
    return np.log(A + 1e-300)


def initB_arch_cover(lmbd, n_st, cover_1k, cover_nd):
    """Computes full Poisson emission probability matrix (reference implementation)."""
    
    # 1. Define lambdas (means)
    mean_n = lmbd[1] * cover_nd
    mean_n2 = lmbd[1] * cover_1k
    mean_af = lmbd[2] * cover_1k
    mean_i2 = lmbd[0] * cover_nd
    
    # 2. Helper function to generate probability vectors
    def get_prob_vec(mu):
        k = np.arange(1, n_st)  # Indices from 1 to n_st-1
        probs = poisson.pmf(k, mu)  # Vectorized Poisson calculation
        p0 = 1.0 - np.sum(probs)  # Residual mass for index 0
        return np.concatenate(([p0], probs))
    
    # 3. Generate vectors
    Paf = get_prob_vec(mean_af)
    Pn = get_prob_vec(mean_n)
    Pn2 = get_prob_vec(mean_n2)
    Pi2 = get_prob_vec(mean_i2)
    
    # 4. Construct emission matrix
    B = np.empty((2, n_st, n_st))
    B[0] = np.outer(Paf, Pn)
    B[1] = np.outer(Pn2, Pi2)
    
    return B

# в функцию подается массив гаплотипов
# k (O1 и О2) – наблюдение, в нашем случае количество мутаций
# от стандартной формулы пуассона берем log:
#   P(k|λ)   =  (λ^k     * e^{-λ}) /  (k!) -> 
# log P(k|λ) = k*log(λ)  -   λ     - log(k!)
# log P(O1=k1, O2=k2 | state) = log P(O1=k1 | state) + log P(O2=k2 | state)
# предподсчитываем все эмиссии
def compute_emissions_custom(O1, O2, L1, L2, rates):
    """Computes vectorized log-emission scores for all haplotypes using simplified Poisson model."""
    M, N = O1.shape  # M - количество гаплотипов (удвоенное колво людей), N - число окон по геному
    n_states = 3
    
    # Output matrix (M x N x 3 states)
    log_emit = np.zeros((M, N, n_states))
    
    # Extract rate parameters
    rate_n = rates[0]
    rate_af = rates[1]
    rate_i_old = rates[2]
    rate_i_young = rates[3]
    
    # Epsilon to avoid log(0)
    eps = 1e-300
    ln_n = np.log(rate_n + eps)
    ln_af = np.log(rate_af + eps)
    ln_old = np.log(rate_i_old + eps)
    ln_young = np.log(rate_i_young + eps)
    
    # Poisson Score = O * ln(rate) - rate * L (ln(O!) cancels)
    # modern:           o_1*log(λ(t_afr)) - λ(t_afr)        + o_2*log(λ(t)) - λ(t)      
    # archaic1:         o_1*log(λ(t)) - λ(t)        + o_2*log(λ(t_1)) - λ(t_1) 
    log_emit[:, :, 0] = (O1 * ln_af - rate_af * L1) + (O2 * ln_n - rate_n * L2)  # 0 = Modern
    log_emit[:, :, 1] = (O1 * ln_n - rate_n * L1) + (O2 * ln_old - rate_i_old * L2)  # 1 = Archaic_old
    log_emit[:, :, 2] = (O1 * ln_n - rate_n * L1) + (O2 * ln_young - rate_i_young * L2) # 2 = Archaic_young
    
    return log_emit


# VITERBI ALGORITHM
@jit(nopython=True, parallel=True)
def viterbi_fast(log_emit, log_trans, log_start):
    """Executes Viterbi algorithm in parallel using Numba to find most likely state path."""
    M, N, n_states = log_emit.shape
    paths = np.zeros((M, N), dtype=np.int32)
    
    # Parallel loop over all haplotypes
    for m in prange(M):
        # Local buffers for dynamic programming
        viterbi = np.zeros((N, n_states))
        backpointer = np.zeros((N, n_states), dtype=np.int32)
        
        # 1. Initialization
        # v_1(j) = π_j*b_j(o_1), b_j - это эмиссии 
        for s in range(n_states):
            viterbi[0, s] = log_start[s] + log_emit[m, 0, s]
        
        # 2. Forward Pass
        # viterbi - таблица значений (рекурсия в процессе) 
        # v_t​(j) = max​(v_{t−1}​(i)      * a_i      * ​b_j​(o_t​)) 
        # =        max(log(v_{t−1}​(i)) + log(a_i) + log(​b_j​(o_t​))). log(​b_j​(o_t​)) не зависит от i => выносим за max
        # =        max(log(v_{t−1}​(i)) + log(a_i)) + log(​b_j​(o_t​)). max_val = max(log(v_{t−1}​(i)) + log(a_i))
        # =        max_val + log(​b_j​(o_t​)) = max_val + log_emit 
        for i in range(1, N):
            for s in range(n_states):
                # Find max(prev_prob + transition)
                max_val = -1e200
                best_prev = 0
                for p in range(n_states):
                    val = viterbi[i - 1, p] + log_trans[p, s]
                    if val > max_val:
                        max_val = val
                        best_prev = p
                
                # Add emission score for current window
                viterbi[i, s] = max_val + log_emit[m, i, s]
                backpointer[i, s] = best_prev
        
        # 3. Backtrace
        # Decide final state, выбор argmax по всем 3 состояниям
        best_last = 0
        best_val = viterbi[N-1, 0]
        for s in range(1, n_states):
            if viterbi[N-1, s] > best_val:
                best_val = viterbi[N-1, s]
                best_last = s
        paths[m, N-1] = best_last
        
        # Reconstruct path backwards
        for i in range(N - 2, -1, -1):
            paths[m, i] = backpointer[i + 1, paths[m, i + 1]]
    
    return paths

# rates = [lambda_n, lambda_af, lambda_old, lambda_young]
# transition_params = [t_old, t_young, a_old, a_young]
def run_hmm(O1, O2, L1, L2, rates, rr, transition_params=[], A=None, pi=None):
    """Orchestrates HMM pipeline: emissions, transitions, and Viterbi."""
    
    print("Calculating emission scores...")
    log_emissions = compute_emissions_custom(O1, O2, L1, L2, rates) # считаем эмиссии (по кол-ву мутаций до O_max, не по всем окнам)
    
    # Transitions
    # Initial probabilities: log_start = np.log(np.array([1 - a_old - a_young, a_old, a_young]) + 1e-300)
    if A is None or pi is None:
        log_A = get_log_A3(1000, rr, transition_params) # считаем переходы
        log_start = np.log(np.array([1 - transition_params[2] - transition_params[3], transition_params[2], transition_params[3]]) + 1e-300)
    else: 
        log_A = np.log(A + 1e-300)
        log_start = np.log(pi + 1e-300)
    
    print("Running Viterbi...")
    paths = viterbi_fast(log_emissions, log_A, log_start)
    
    return paths

def get_tracts(vector, step=1000):
    """Converts binary state vector into dictionary of genomic intervals (start, end)."""
    # 0 = modern
    # 1 = archaic_old
    # 2 = archaic_young
    result = {0: [], 1: [], 2: []}
    
    current_state = vector[0]
    start_index = 0
    
    for i, state in enumerate(vector):
        if state != current_state:
            result[current_state].append((start_index * step, i * step - 1)) # [start, end]
            current_state = state
            start_index = i
    
    result[current_state].append((start_index * step, len(vector) * step - 1))
    
    return {
        "Modern": result[0],
        "Archaic_old": result[1],
        "Archaic_young": result[2]
    }


def clean_gaps(dct, gap_file, target_chrom):
    """Filters genomic regions defined in gap file from inferred tracts."""
    
    print('Processing gaps...')
    raw_gaps = []
    
    try:
        with open(gap_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[1] == 'chr' + target_chrom:
                    raw_gaps.append((int(parts[2]), int(parts[3]) - 1))
    except FileNotFoundError:
        print(" !!! Gap file not found.")
        return dct
    

    
    # Merge overlapping gaps
    merged_gaps = []
    if raw_gaps:
        raw_gaps.sort()
        merged_gaps = [raw_gaps[0]]
        for curr in raw_gaps[1:]:
            prev = merged_gaps[-1]
            if curr[0] <= prev[1] + 1:
                merged_gaps[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged_gaps.append(curr)
    
    # Helper to subtract gaps from interval
    def subtract(interval, gaps):
        start, end = interval
        res = []
        curr = start
        for g_s, g_e in gaps:
            if g_e < curr:
                continue
            if g_s > end:
                break
            if curr < g_s:
                res.append((curr, g_s - 1))
            curr = max(curr, g_e + 1)
        if curr <= end:
            res.append((curr, end))
        return res
    
    # Process dictionary
    new_dct = {}
    for sample, categories in dct.items():
        new_dct[sample] = {}
        for cat, intervals in categories.items():
            cleaned_list = []
            for interval in intervals:
                if not merged_gaps:
                    cleaned_list.append(interval)
                else:
                    cleaned_list.extend(subtract(interval, merged_gaps))
            new_dct[sample][cat] = cleaned_list
    
    return new_dct


def run_daiseg(json_file): # type: ignore
    """Main pipeline: runs HMM and saves ONLY TSV output."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create observation sequences
    tsv_path = data["prefix"] + '/' + data["data"]
    bed_path = data["prefix"] + '/' + data["window_callability"]["Thousand_genomes"]
    
    obs_seq, nst = create_observations(tsv_path, bed_path)
    
    if len(obs_seq) > 0:
        first_key = list(obs_seq.keys())[0]
#        print(f"Sample length: {len(obs_seq[first_key])}")

    
    # Extract parameters
    prms = data["parameters_initial"]
    gen_time = prms['generation_time']
    mu = prms['mutation']
    rr = prms['rr']
    l = prms['window_length']

    # зачем мы делим на ген_тайм если только что до этого умножали на него? 
    d = mu * l / gen_time
    lambda_0 = [
        d * prms['t_archaic_c'],
        d * prms['t_split_c'],
        d * prms['t_introgression_old_c'],
        d * prms['t_introgression_young_c']
    ]
    transition_params = [
        prms['t_introgression_old'] / gen_time,
        prms['t_introgression_young'] / gen_time,
        prms['admixture_proportion_old'],
        prms['admixture_proportion_young']
    ]

    
    # Load callability files
    cal_1kG = np.loadtxt(data['prefix'] + '/' + data["window_callability"]["Thousand_genomes"], usecols=-1)
    cal_nd_1kG = np.loadtxt(data['prefix'] + '/' + data["window_callability"]["Nd_1k_genomes"], usecols=-1)
    
    # Prepare matrices
    O1, O2, names = prepare_matrices_from_dict(obs_seq)
    
    # Run HMM
    result = run_hmm(O1, O2, cal_1kG, cal_nd_1kG, lambda_0, rr, transition_params=transition_params)
    dictionary = {k: v for k, v in zip(names, result)}
    
    # Extract tracts
    out_dict = {}
    for name in names:
        out_dict[name] = get_tracts(dictionary[name])
    
    # Remove gaps
    out_dict_new = clean_gaps(out_dict, data["gaps"], data["CHROM"])
    
    # Save TSV results
    output_tsv = f"{data['prefix']}/{data['output']}.tsv"
    print(f" Saving TSV results to: {output_tsv}")
    
    rows = []
    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("Sample\tCHROM\tStart\tEnd\tLength\tState\n")
        for sample_name, tracks in out_dict_new.items():
            for state_name, intervals in tracks.items():
                for start, end in intervals:
                    f.write(f"{sample_name}\t{data['CHROM']}\t{start}\t{end}\t{end-start+1}\t{state_name}\n")
                    rows.append({
                        "Sample": sample_name,
                        "CHR": data['CHROM'],
                        "Start": start,
                        "End": end,
                        "State": state_name
                    })
    
    df_result = pd.DataFrame(rows)
    return df_result, out_dict_new


# Store original logic for reference
_original_logic = run_daiseg


def _worker_proxy(filepath):
    """Helper function for pickle compatibility (must be defined globally)."""
    return _original_logic(filepath)


def run_daiseg(json_input):
    """
    Wrapper: handles single file (string) or list of files (parallel).
    """

    # Single file → run normally БЕЗ multiprocessing
    if not isinstance(json_input, list):
        print(f" Processing single file (no parallelization needed)...")
        return _original_logic(json_input)

    # Single file in list → тоже без multiprocessing
    if len(json_input) == 1:
        print(f" Processing single file (sequential)...")
        return [_original_logic(json_input[0])]

    # Уже в подпроцессе → выполнять последовательно
    if multiprocessing.current_process().daemon:
        return [_original_logic(f) for f in json_input]

    # Множество файлов → параллельно
    MAX_WORKERS = 64
    cpu_count = multiprocessing.cpu_count()
    pool_size = min(cpu_count - 1, MAX_WORKERS, len(json_input))
    pool_size = max(pool_size, 1)
    
    print(f" Parallelizing {len(json_input)} files on {pool_size} cores...")

    with multiprocessing.Pool(processes=pool_size) as pool:
        return pool.map(_worker_proxy, json_input)
