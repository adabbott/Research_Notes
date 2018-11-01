import os

os.chdir("PES_data")

for i in range(1,181):
    os.chdir(str(i))
    print("in" + str(i))
    os.system("psi4 input.dat")
    os.chdir("../")
