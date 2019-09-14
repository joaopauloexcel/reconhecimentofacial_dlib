import cv2
import time
import os
import glob #serve para percorrer uma pasta
import _pickle as cPickle
import dlib
import numpy as np
import treinamento as treino

def limpar():
    pasta = "fotos/treinamento"
    diretorio = os.listdir(pasta)
    for arquivo in diretorio:
        os.remove('{}/{}'.format(pasta, arquivo))

totalFotos=3
nome="joao"
senha="123"

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()#detecto de treinamento de face já incluso no código do dlib
#treinamento do dlib retornar os 68 pontos terminantes das faces)
emLoop = True
fhotos = 0
while (emLoop):
    ret, foto = cap.read()
    frame = cv2.flip(foto, 2)
    cv2.putText(frame, "Foto {}/{}".format(fhotos + 1, totalFotos), (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (0, 255, 255))
    k = cv2.waitKey(100)
    if k == 27:
         emLoop = False
    elif k == ord('s'):
        frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # passa imagem para tom de cinza
        facesDetectadas = detector(frameCinza, 1)  # aplica imagem em cinza no detector de faces do dlib
        if facesDetectadas:
            cv2.imwrite("fotos/treinamento/{}_{}_{}.jpg".format(nome, senha, fhotos + 1), frame)
            fhotos += 1
            if fhotos == totalFotos:
                treino.treino(senha)
                limpar()
                emLoop = False
    cv2.imshow('Gravacao', frame)
    if cv2.waitKey(1) == ord('q'):
        emLoop = False
cap.release()
cv2.destroyAllWindows()
