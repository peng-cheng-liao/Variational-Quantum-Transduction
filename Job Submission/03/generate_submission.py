import numpy as np

from main import *

name = "Transduction_binomial_code"


def generate_submission():
    submissionfile = 'PythonsAndJobs/submission_' + name + '.job'
    with open(submissionfile, 'w') as script_file:
        script_file.write('#!/bin/bash\n'
                          '#SBATCH --account=qzhuang_922\n'
                          '#SBATCH --partition=epyc-64\n\n')

        nlist = np.around(np.linspace(0.0, 10, 10),2)
        for n in nlist:
            script_file.write(f'sbatch e={n}.job\n')
            # script_file.write('sbatch ' + name + '_epsilon=%s.job\n' % epsilon)


generate_submission()
