initial_number = 469182


string = "scancel "
#string = "rm "
for i in range(20):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
