initial_number = 454560


string = "scancel "
#string = "rm "
for i in range(200):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
