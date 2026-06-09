initial_number = 6205654


string = "scancel "
#string = "rm "
for i in range(3600):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
