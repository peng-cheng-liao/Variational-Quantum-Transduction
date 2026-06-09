initial_number = 4523722


string = "scancel "
#string = "rm "
for i in range(50):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
