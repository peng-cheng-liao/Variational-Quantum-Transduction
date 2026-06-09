initial_number = 1735001


string = "scancel "
#string = "rm "
for i in range(800):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
