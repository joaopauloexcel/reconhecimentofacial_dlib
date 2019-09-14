import cv2
import time
import os
import glob #serve para percorrer uma pasta
import _pickle as cPickle
import dlib
import numpy as np
import treinamento as treino

totalFotos=3
nome="thiago"
senha="123"

def imprimePontos(imagem, pontosFaciais):#faz 68 circulos em faces detectadas
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()#detecto de treinamento de face já incluso no código do dlib
#treinamento do dlib retornar os 68 pontos terminantes das faces)
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")

emLoop = True
fhotos = 0
while (emLoop):
    ret, frame = cap.read()
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#passa imagem para tom de cinza
    facesDetectadas = detector(frameCinza, 1)#aplica imagem em cinza no detector de faces do dlib

    for face in facesDetectadas:#percorre a matrix dos pontos das faces localizadas
        #atribuição dos pontos correspondentes ao limite esquerdo, direito, top e base da face detectada
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        #exibe imagem com retangulo verde
        cv2.rectangle(frame, (e, t), (d, b), (0, 255, 0), 2)
        # percorre a área da matiz de faces presentes na imagem original
        for face in facesDetectadas:
            # Recebe os parâmetros da escala x,y dos pontos das faces
            pontos = detectorPontos(frame, face)
            print(pontos.parts())
            print(len(pontos.parts()))
            imprimePontos(frame, pontos)  # contorna a face na tela com 68 circulos
            cv2.putText(frame, "Foto {}/{}".format(fhotos+1,totalFotos), (e + 10, b + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255))
            k = cv2.waitKey(100)
            if k == 27:
                emLoop = False
            elif k == ord('s'):
                cv2.imwrite("fotos/treinamento/{}_{}_{}.jpg".format(nome, senha, fhotos + 1), frame)
                fhotos += 1
                if fhotos == totalFotos:
                    treino.treino(senha)
                    emLoop = False
    cv2.imshow('Gravacao', frame)
    if cv2.waitKey(1) == ord('q'):
        emLoop = False
cap.release()
cv2.destroyAllWindows()
