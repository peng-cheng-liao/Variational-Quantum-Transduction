import numpy as np


nlist = np.around(np.linspace(0.0, 10, 10),2)
original_file = 'source.job'
for n in nlist:
        new_file = f'PythonsAndJobs/e={n}.job'
        line_to_replace = "python e=1.0.py"
        new_line_content = f"python e={n}.py"


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





