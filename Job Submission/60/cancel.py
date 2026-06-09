initial_number = 4105081


string = "scancel "
#string = "rm "
for i in range(8000):
    number = initial_number + i
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "%s" % number + " "
print(string)
