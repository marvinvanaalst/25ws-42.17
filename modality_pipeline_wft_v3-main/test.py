a = {'lola': 1, 'lolo': 2, 'lili': 3, 'mbape': 4}

def soma(lola, lolo, lili, **extra):
    print( lola + lolo + lili)
    print(extra)

soma(**a)