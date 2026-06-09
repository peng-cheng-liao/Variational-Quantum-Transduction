initial_number = 23528207


string = "scancel "
#string = "rm "
for i in range(500):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
