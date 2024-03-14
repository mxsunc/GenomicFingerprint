import numpy as np
from Bio.Seq import Seq

#helper function for generating a string of nucleotides of any length
def gen_ran_nuc(len_nuc):
    return ''.join(np.random.choice(['A', 'T', 'C', 'G'], len_nuc))

#generate a variant (5p, 3p, ref, alt, chromosome, position, strand) with a certain chance of being an indel
def generate_variant(length=6, indel_percent=.1):
    assert length >= 2
    assert length % 2 == 0
    five_p = gen_ran_nuc(length)
    three_p = gen_ran_nuc(length)
    choices = ['A', 'T', 'C', 'G']
    if np.random.sample() < indel_percent:
        size = int(np.random.choice(range(1, length + 1), 1))
        ##even chance of being insertion vs. deletion
        if np.random.sample() < .5:
            ref = '-' * length
            alt = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
        else:
            ref = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
            alt = '-' * length
    else:
        ref = ''.join(np.random.choice(choices, 1))
        remaining_choices = choices.copy()
        remaining_choices.pop(choices.index(ref))
        alt = ''.join(np.random.choice(remaining_choices, 1))
        ref += '-' * (length - 1)
        alt += '-' * (length - 1)
    chromosome = np.random.choice(range(1, 25))
    position = np.random.sample()
    strand = np.random.choice([1, 2])
    return np.array(list(five_p)), np.array(list(three_p)), np.array(list(ref)), np.array(list(alt)), chromosome, position, strand

##to make sure the reverse of a simulated variant
def check_variant(variant, positive_variants):
    five_p = ''.join(variant[0])
    three_p = ''.join(variant[1])
    ref = ''.join(variant[2])
    alt = ''.join(variant[3])

    x = False
    for pos in positive_variants:
        if five_p == ''.join(pos[0]) and three_p == ''.join(pos[1]) and ref == ''.join(pos[2]) and alt == ''.join(pos[3]):
            x = True
            break
    if x:
        return x
    else:
        five_p_rev = str(Seq.reverse_complement(Seq(five_p)))
        three_p_rev = str(Seq.reverse_complement(Seq(three_p)))
        ref_rev = str(Seq.reverse_complement(Seq(ref.replace('-',''))))
        alt_rev = str(Seq.reverse_complement(Seq(alt.replace('-',''))))
        for pos in positive_variants:
            if five_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[1])))) and three_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[0])))) and ref_rev == str(Seq.reverse_complement(Seq(''.join(pos[2]).replace('-','')))) and alt_rev == str(Seq.reverse_complement(Seq(''.join(pos[3]).replace('-', '')))):
                x = True
                break
    return x

def generate_survival_sample(risk_variants, min_variants=1, max_variants=500, peak_variants=50, std_variants=0.75, base_survival_time=365, survival_time_std=100):
    # Generate total variants using a log-normal distribution
    mu = np.log(peak_variants)  # Convert peak to log-scale for mu parameter
    total_variants = np.random.lognormal(mean=mu, sigma=std_variants, size=1)
    total_variants = int(np.clip(total_variants, min_variants, max_variants))
    
    risk_variant_proportion = np.random.randint(0, 10) * 0.01
    num_risk_variants = int(total_variants * risk_variant_proportion)
    
    variants = [generate_variant() for _ in range(total_variants - num_risk_variants)]
    variant_indices = np.random.choice(len(risk_variants), size=num_risk_variants, replace=True)
    variants += [risk_variants[i] for i in variant_indices]
    
    np.random.shuffle(variants)
    
    survival_time = np.random.normal(base_survival_time, survival_time_std)
    risk_level = 'High Risk' if risk_variant_proportion > 0.05 else 'Low Risk'
    
    # Increase the effect of risk variants
    event_observed = True
    if risk_level == 'High Risk':
        survival_time *= np.random.uniform(0.3, 0.5)  # More significant reduction for high risk
    else:
        # For some low-risk patients, set event_observed to False to indicate they do not die
        if np.random.rand() < 0.5:  # Adjust this probability as needed
            event_observed = False
        else:
            event_observed = True
    
    # Optionally, for low-risk patients, adjust survival time to be higher on average
    if risk_level == 'Low Risk' and event_observed:
        survival_time *= np.random.uniform(1.1, 1.3)  # Increase survival time for some low-risk patients

    return variants, survival_time, event_observed, risk_level
