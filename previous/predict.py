import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.fast_predictor as fp

if __name__ == "__main__" :  
    params = { "model_dir" : "/Storage/Otonno_2019/CC6204_Deep_Learning/Tarea_1PyC/modelsQuick/Adam_0_001_sigmoid",
               "data_dir" : "/Storage/Otonno_2019/CC6204_Deep_Learning/Tarea_1PyC/datosQUICKDRAW",
               "device" : "/gpu:0",                
               "number_of_classes" : 12,
               "modelo" : 'Adam_0_001_sigmoid'
          }
    
#     params = { "device" : "/gpu:0",
#               "model_dir" : "/home/vision/smb-datasets/MNIST-small/models",
#               "data_dir" : "/home/vision/smb-datasets/MNIST-small",                          
#               "number_of_classes" : 10
#         }
    print(os.getcwd())
    # os.chdir('datosQUICKDRAW/')

    map = open('datosQUICKDRAW/mapping.txt', 'r')
    mapa = {}
    for ln in map:
        nombre, label = ln.split()
        mapa[nombre] = label
        print(mapa)

    predictor = fp.FastPredictor(params)
    imagenes = os.listdir("/".join([params["data_dir"],"test_images"]))
    file = open(os.path.join(params["model_dir"], "test"),"w+")
    file.write("path_imagen, label, prediction, probabilidad\n")
    print("Archivo para escritura creado")
    print("prediciendo")
    for dir in imagenes:
        et = dir
        dir = os.listdir("/".join([params["data_dir"], "test_images", dir]))
        for img in dir:
            if '.jpg' in img:
                label = mapa[et]
                img = "/".join([params["data_dir"], "test_images", et, img])
                prediction = predictor.predict(img)
                frase = (img + ", %i, %i, %.5f\r\n") % (int(label), prediction[0], prediction[1])
                file.write(frase)
                print(frase)
    file.close()
    # while True :
    #     filename = input("Image: ")
    #     prediction = predictor.predict(filename)
    #     print(prediction)
    #
        