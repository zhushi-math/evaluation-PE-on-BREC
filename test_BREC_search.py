import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PE_METHODS = [
    # "adj_powers",
    # "rrwp",
    # "bernstein",
    # "bern_mixed_sym2",
    # "bern_mixed_sym3",
    # "bern_mixed_smooth",
    #"SPD",
    #"resistance_distance",
    "bern_SPD",
    "bern_resistance_distance",
]

PE_POWERs = list(range(4, 28, 2))

for pe_method in PE_METHODS:
    for pe_power in PE_POWERs:
        for seed in SEED:
            script_base = 'python test_BREC.py --config configs/BREC.json --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4'
            script_base += f' --BATCH_SIZE=32 --SEED={seed} --PE_METHOD {pe_method} --PE_POWER {pe_power}'
            print(script_base)
            os.system(script_base)
