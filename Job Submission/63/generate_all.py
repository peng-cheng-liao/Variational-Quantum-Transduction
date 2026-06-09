import numpy as np

# replicate python files
#etalist = np.around(np.arange(0.31, 0.4, 0.01), 2)
#etalist = np.around(np.arange(0.01, 0.1, 0.01),2)
etalist1 = np.around(np.arange(0.05, 1.0, 0.05), 2)
etalist2 = np.around(np.arange(0.31, 0.4, 0.01), 2)
etalist = np.sort(np.concatenate((etalist1, etalist2)))
randomizationlist = np.arange(20)
d1_list = np.arange(2, 3, 1)
d2_list = np.arange(1, 2, 1)


def replicate_py_file():
    original_file = 'source.py'
    for eta in etalist:
        for randomization in randomizationlist:
            for d1 in d1_list:
                for d2 in d2_list:
                    new_file = f'PythonsAndJobs/eta={eta}_d1={d1}_d2={d2}_randomization={randomization}.py'
                    nline_to_replace = "eta = 0.1"
                    new_nline_content = f"eta = {eta}"

                    rline_to_replace = "randomization = 0"
                    new_rline_content = f"randomization = {randomization}"

                    d1line_to_replace = "d1 = 2"
                    new_d1line_content = f"d1 = {d1}"

                    d2line_to_replace = "d2 = 1"
                    new_d2line_content = f"d2 = {d2}"

                    # Read the original file
                    with open(original_file, 'r') as file:
                        lines = file.readlines()

                    # Find and replace the line
                    for i, line in enumerate(lines):
                        if line.strip() == nline_to_replace:
                            lines[i] = new_nline_content + '\n'
                            break  # Exit the loop after the first occurrence is replaced

                    for i, line in enumerate(lines):
                        if line.strip() == rline_to_replace:
                            lines[i] = new_rline_content + '\n'
                            break  # Exit the loop after the first occurrence is replaced
                    for i, line in enumerate(lines):
                        if line.strip() == d1line_to_replace:
                            lines[i] = new_d1line_content + '\n'
                            break  # Exit the loop after the first occurrence is replaced
                    for i, line in enumerate(lines):
                        if line.strip() == d2line_to_replace:
                            lines[i] = new_d2line_content + '\n'
                            break  # Exit the loop after the first occurrence is replaced

                    # Write the modified content to a new file
                    with open(new_file, 'w') as file:
                        file.writelines(lines)

                    print("Line replaced and saved as", new_file)


# replicate job files
def replicate_job_file():
    original_file = 'source.job'
    for eta in etalist:
        for r in randomizationlist:
            for d1 in d1_list:
                for d2 in d2_list:
                    new_file = f'PythonsAndJobs/eta={eta}_d1={d1}_d2={d2}_randomization={r}.job'
                    line_to_replace = "python eta=0.1_d1=2_d2=1_randomization=0.py"
                    new_line_content = f"python eta={eta}_d1={d1}_d2={d2}_randomization={r}.py"

                    # Read the original file
                    with open(original_file, 'r') as file:
                        lines = file.readlines()

                    # Find and replace the line
                    for i, line in enumerate(lines):
                        if line.strip() == line_to_replace:
                            lines[i] = new_line_content + '\n'
                            break  # Exit the loop after the first occurrence is replaced

                    # Write the modified content to a new file
                    with open(new_file, 'w') as file:
                        file.writelines(lines)

                    print("Line replaced and saved as", new_file)



def generate_submission():
    submissionfile = 'PythonsAndJobs/submission.job'
    with open(submissionfile, 'w') as script_file:
        script_file.write('#!/bin/bash\n'
                          '#SBATCH --account=qzhuang_922\n'
                          '#SBATCH --partition=epyc-64\n\n')

        for d1 in d1_list:
            for d2 in d2_list:
                for eta in etalist:
                    for r in randomizationlist:
                        script_file.write(f'sbatch eta={eta}_d1={d1}_d2={d2}_randomization={r}.job\n')


replicate_py_file()
replicate_job_file()
generate_submission()
