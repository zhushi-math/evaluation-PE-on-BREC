from .adj_powers import compute_adjacency_power_series
from .rrwp import compute_rrwp
from .bernstein import compute_bernstein_polynomials
from .bern_mixed_sym2 import compute_bern_mixed_sym2
from .bern_mixed_sym3 import compute_bern_mixed_sym3
from .bern_mixed_smooth import compute_bern_mixed_smooth
from .SPD import SPD_floyd_warshall
from .resistance_distance import resistance_distance
from .bern_SPD import compute_bernstein_polynomials_SPD
from .bern_resistance_distance import  compute_bernstein_polynomials_resistance_distance


pe_computer_dict = {
    "adj_powers": compute_adjacency_power_series,       # output_len: K+1
    "rrwp": compute_rrwp,                               # output_len: K+1
    "bernstein": compute_bernstein_polynomials,         # output_len: K+2
    "bern_mixed_sym2": compute_bern_mixed_sym2,         # output_len: K*2+1
    "bern_mixed_sym3": compute_bern_mixed_sym3,         # output_len: K//2*5+1
    "bern_mixed_smooth": compute_bern_mixed_smooth,     # output_len: K+1
    "SPD": SPD_floyd_warshall,                          
    "resistance_distance": resistance_distance,
    "bern_SPD": compute_bernstein_polynomials_SPD,
    "bern_resistance_distance": compute_bernstein_polynomials_resistance_distance,
}
