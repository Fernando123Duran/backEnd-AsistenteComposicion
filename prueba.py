import onnxruntime as ort 
import numpy as np
from scipy.special import softmax
duration=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]
ligas=[19,20,21]
unSoloValor=[0,1,2,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,]

parte=[ 51, 47, 40, 112, 4, 116, 6, 116, 4, 19, 116, 4, 21, 116, 6, 114, 4, 116, 6, 114, 6, 116, 4, 114, 4, 116, 4, 118, 4, 119, 9, 26, 6, 111, 4, 114,
      6, 114, 4, 19, 114, 4, 21, 114, 6, 116, 4, 114, 7, 109, 4, 111, 4, 109, 4, 108, 4, 111, 4, 109, 8, 26, 8, 30, 29, 109, 4, 112, 6, 112, 4, 19, 112, 4, 
21, 112, 6, 111, 4, 112, 6, 111, 6, 112, 4, 111, 4, 112, 4, 114, 4, 116, 9, 26, 6, 111, 4, 114, 6, 114, 4, 19, 114, 4, 21, 114, 6, 116, 4, 114, 7, 111, 
4, 111, 4, 109, 4, 108, 4, 111, 4, 22, 27, 109, 9, 26, 6, 30, 23, 27, 24, 109, 7, 116, 4, 19, 116, 4, 21, 116, 6, 114, 4, 29, 116, 4, 116, 6, 114, 4, 116,
7, 114, 4, 116, 9, 26, 6, 27, 25, 111, 4, 114, 6, 114, 4,]
def generar(parte):
    
    nuevaList=[]
    # print(part)
    ort_session = ort.InferenceSession('models/Epoch50v2.onnx')
    for i in range(3):
        part=np.array(parte)
        part=np.expand_dims(part,axis=(0))
        part=np.int64(part)
        salida= ort_session.run(["output"],{"input":part})[0]
        salida=salida[:,-1,:]/1
        probs =softmax(salida,axis=-1)
        n=np.argmax(probs,axis=-1)
        print(n[0])
        if len(nuevaList)==2 and n[0] in ligas:
            nuevaList.append(n[0])
        if n[0]in unSoloValor:
            nuevaList.append(n[0])
            break
        if n[0] in duration or n[0] in range(58,145+1) :
            if len(nuevaList)!=2: 
                nuevaList.append(n[0])
        parte=np.concatenate((parte,nuevaList))
    print(nuevaList)  

    # otra=np.concatenate((parte,[1,4,5,6]))
    # print(n)
    # print(otra)

if __name__ == '__main__':
    # generar(parte)
    per=[1,2,3,4,5,6,7,8,9,10]
    por=per[-5:]
    print(por)