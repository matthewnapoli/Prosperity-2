table = [
    [1.0,0.48,1.52,0.71],
    [2.05,1.0,3.26,1.56],
    [0.64,0.3,1,0.46],
    [1.41, 0.61, 2.08, 1.0]
    ]


possibilites = []

for trad in range(0,4):
    for trad2 in range(0,4):
        for trad3 in range(0,4):
            for trad4 in range(0,4):
                start = [1.0,""]
                start[0] = table[3][trad]*table[trad][trad2]*table[trad2][trad3]*table[trad3][trad4]*table[trad4][3]
                start[1] = "4"+str(trad+1)+str(trad2+1)+str(trad3+1)+str(trad4+1)+"4"
                possibilites.append(start)
                print(start)


maxarr = -1.0
idxmax = -1.0
for i in range(len(possibilites)):
    if possibilites[i][0] > maxarr:
        maxarr = possibilites[i][0]
        idxmax = i


print(maxarr)
print(idxmax)
print(possibilites[idxmax][1])