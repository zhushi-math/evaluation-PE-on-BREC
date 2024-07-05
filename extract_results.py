import os
import numpy as np


folder = "result_BREC"

with open('summary.txt', 'w') as wfile:

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('txt'):
            continue
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as rfile:
            lines = rfile.readlines()
            if len(lines) != 20:
                continue

            results = []
            for idx in range(1, 21, 2):
                line = lines[idx]
                num_corrects = int(line.split('\t', 1)[0])
                results.append(num_corrects)

            print(filename, file=wfile)
            print('avg:', np.mean(results), file=wfile)
            print('std:', np.std(results), file=wfile)
            print('max:', np.max(results), file=wfile)
            print('min:', np.min(results), file=wfile)
            print('', file=wfile)
