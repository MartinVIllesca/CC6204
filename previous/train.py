"""
Author: jose.saavedra
This is an example for training on a sketch classification problem
model_dir and data_dir should changed to the correct paths
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.mlp as mlp

if __name__ == "__main__" :
    params = { "device" : "/gpu:0",
              "model_dir" : "/Storage/Otonno_2019/CC6204_Deep_Learning/Tarea_1PyC/modelsQuick/Adam_0_001_sigmoid",
              "data_dir" : "/Storage/Otonno_2019/CC6204_Deep_Learning/Tarea_1PyC/datosQUICKDRAW/",
              "learning_rate" : 0.001,
              "number_of_classes" : 12,
              "number_of_iterations" : 40000,
              "batch_size" : 80,
              "data_size" : 12000,
              "activation_function": 'sigmoid',
              "optimizer": 'Adam_0_001' # Adam or GradientDescent
        }
    my_mlp = mlp.MLP(params)
    print("MLP initialized ok")
    print("--------start training")
    # my_mlp.train()
    my_mlp.save_model()
    # Use my_mlp.test() for testing
    #Use my_mlp.save_model() for saving models that will be used for fast_prediction
    print("--------end training")
    
