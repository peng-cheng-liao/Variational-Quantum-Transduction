import numpy as np

etalist = np.around(np.arange(0.1, 1.0, 0.1),1)
randomizationlist = np.arange(20)
etalist = np.around(np.arange(0.1, 0.2, 0.1),1)

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





