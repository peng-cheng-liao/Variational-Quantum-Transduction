import numpy as np

etalist = np.around(np.arange(0.1, 1.0, 0.1),1)
etalist = np.around(np.arange(0.1, 0.2, 0.1),1)
randomizationlist = np.arange(20)
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



