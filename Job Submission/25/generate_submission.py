import numpy as np

from main import *

name = "Transduction_binomial_code"

etalist = np.around(np.arange(0.1, 1.0, 0.1),1)
randomizationlist = np.arange(20)



def generate_submission():
    submissionfile = 'PythonsAndJobs/submission.job'
    with open(submissionfile, 'w') as script_file:
        script_file.write('#!/bin/bash\n'
                          '#SBATCH --account=qzhuang_922\n'
                          '#SBATCH --partition=epyc-64\n\n')


        for eta in etalist:
            for randomization in randomizationlist:
                script_file.write(f'sbatch eta={eta}_randomization={randomization}.job\n')
            # script_file.write('sbatch ' + name + '_epsilon=%s.job\n' % epsilon)


generate_submission()
