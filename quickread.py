
import os

os.chdir('datosQUICKDRAW/')

map = open('mapping.txt', 'r')
mapa = {}
for ln in map:
    nombre, label = ln.split()
    mapa[nombre] = label

f = open('test.txt','w')

etiqueta = os.listdir(path='test_images/')

for et in etiqueta:
    imgs = os.listdir(path='test_images/' + et)
    for im in imgs:
        direccion = os.getcwd() + '/test_images/' + et + '/' + im + '\t' + mapa[et]
        f.write(direccion + '\n')

# f = open('test.txt','w')
f.close()