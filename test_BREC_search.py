import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PE_METHODS = ["bern_poly", "rrwp", "adj_powers", "mixed_bern_poly", "deco_bern_poly"]
PE_LENs = [6, 8, 10, 12, 14]

for pe_method in PE_METHODS:
    for pe_len in PE_LENs:
        for seed in SEED:
            script_base = 'python test_BREC.py --config configs/BREC.json --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4'
            script_base += f' --BATCH_SIZE=32 --SEED={seed} --PE_METHOD {pe_method} --PE_LEN {pe_len}'
            print(script_base)
            os.system(script_base)


PE_LENs = [16, 18]

for pe_method in PE_METHODS:
    for pe_len in PE_LENs:
        for seed in SEED:
            script_base = 'python test_BREC.py --config configs/BREC.json --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4'
            script_base += f' --BATCH_SIZE=32 --SEED={seed} --PE_METHOD {pe_method} --PE_LEN {pe_len}'
            print(script_base)
            os.system(script_base)
