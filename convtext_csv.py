import re
try :
    fname = input("Veuillez Entrer le nom du fichier : ")
    fin = open(fname, "rt")
    fout = open("out.txt", "wt")
except : 
    print("Il y a eu une erreur dans l'ouverture du fichier")
    quit()

for line in fin :
    line = re.sub('\s+',',',line)
    line = line[:-1]
    fout.write(line)
    fout.write('\n')
fin.close()
fout.close()