import numpy as np


# replicate python files
etalist = np.around(np.arange(0.05, 1.0, 0.05),2)
randomizationlist = np.arange(20)



def replicate_py_file():
    original_file = 'source.py'
    for eta in etalist:
        for randomization in randomizationlist:
            new_file = f'PythonsAndJobs/eta={eta}_randomization={randomization}.py'
            nline_to_replace = "eta = 0.1"
            new_nline_content = f"eta = {eta}"
            rline_to_replace = "randomization = 0"
            new_rline_content = f"randomization = {randomization}"

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

            # Write the modified content to a new file
            with open(new_file, 'w') as file:
                file.writelines(lines)

            print("Line replaced and saved as", new_file)



# replicate job files
def replicate_job_file():
    original_file = 'source.job'
    for eta in etalist:
        for randomization in randomizationlist:
            new_file = f'PythonsAndJobs/eta={eta}_randomization={randomization}.job'
            line_to_replace = "python eta=0.1_randomization=0.py"
            new_line_content = f"python eta={eta}_randomization={randomization}.py"

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


        for eta in etalist:
            for randomization in randomizationlist:
                script_file.write(f'sbatch eta={eta}_randomization={randomization}.job\n')

replicate_py_file()
replicate_job_file()
generate_submission()