initial_number = 1384037


string = "scancel "
#string = "rm "
for i in range(180):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
