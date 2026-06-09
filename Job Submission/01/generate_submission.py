import numpy as np

from main import *

name = "Transduction_binomial_code"


def generate_submission():
    submissionfile = 'PythonsAndJobs/submission.job'
    with open(submissionfile, 'w') as script_file:
        script_file.write('#!/bin/bash\n'
                          '#SBATCH --account=qzhuang_922\n'
                          '#SBATCH --partition=epyc-64\n\n')

        nlist = np.arange(0.5, 4.5, 0.5, )
        randomizationlist = np.arange(50)
        for n in nlist:
            for randomization in randomizationlist:
                script_file.write(f'sbatch n={n}_randomization={randomization}.job\n')
            # script_file.write('sbatch ' + name + '_epsilon=%s.job\n' % epsilon)


generate_submission()
