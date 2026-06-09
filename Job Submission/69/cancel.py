initial_number = 5398603


string = "scancel "
#string = "rm "
for i in range(400):
    number = initial_number
    #string = string + " slurm-" + "%s" % number + ".out"
    string = string +  "5399479_%s" % i + " "
print(string)
