import numpy as np

nlist = np.arange(0.5, 4.5, 0.5, )
randomizationlist = np.arange(50)
original_file = 'source.job'
for n in nlist:
    for randomization in randomizationlist:
        new_file = f'PythonsAndJobs/n={n}_randomization={randomization}.job'
        line_to_replace = "python n=1.0_randomization=0.py"
        new_line_content = f"python n={n}_randomization={randomization}.py"


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





