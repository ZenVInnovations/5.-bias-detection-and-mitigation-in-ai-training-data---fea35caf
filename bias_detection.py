import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def distribution_by_attribute(data, attribute):
    '''Calculate the distribution of values for a given attribute.'''
    if attribute not in data.columns:
        raise ValueError(f"Attribute '{attribute}' not found in data.")
    return data[attribute].value_counts(normalize=True)

def chi_square_test(data, attribute, target):
    '''Perform Chi-square test of independence between attribute and target.'''
    if attribute not in data.columns or target not in data.columns:
        raise ValueError(f"Attributes '{attribute}' or '{target}' not found in data.")
    contingency_table = pd.crosstab(data[attribute], data[target])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

def detect_bias(data, sensitive_attributes, target):
    '''Detect bias by checking distribution and statistical dependence.'''
    results = {}
    for attr in sensitive_attributes:
        dist = distribution_by_attribute(data, attr)
        chi2, p = chi_square_test(data, attr, target)
        results[attr] = {
            "distribution": dist.to_dict(),
            "chi2_statistic": chi2,
            "p_value": p,
            "biased": p < 0.05  # significance level
        }
    return results
