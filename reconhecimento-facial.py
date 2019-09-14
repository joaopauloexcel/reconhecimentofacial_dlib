import os
import dlib
import cv2
import numpy as np

def reconhecimentoFacial():
    senha="123"
    cap = cv2.VideoCapture(0)
    detectorFace = dlib.get_frontal_face_detector()
    detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
    reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
    try:
        # Cria um vetor com os indices.@@
        indices = np.load("recursos/indices_{}.pickle".format(senha), allow_pickle=True)
        # Carrega as caracteristicas extraidas das faces treinadas.@@
        descritoresFaciais = np.load("recursos/descritores_{}.npy".format(senha), allow_pickle=True)
    except:
        print('Rede não treinada')
    limiar = 0.5
    while cap.isOpened():
        conectado, video = cap.read()
        frame = cv2.flip(video, 2)
        frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesDetectadas = detectorFace(frameCinza)#aplica imagem em cinza no detector de faces do dlib
        for face in facesDetectadas:
            e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
            pontosFaciais = detectorPontos(frameCinza, face)
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(frame, pontosFaciais)
            listaDescritorFacial = [fd for fd in descritorFacial]
            npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
            distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
            minimo = np.argmin(distancias)
            distanciaMinima = distancias[minimo]
            if distanciaMinima <= limiar:
                nome = os.path.split(indices[minimo])[1].split("_")[0]
                corAlert = {'b': 0, 'g': 255, 'r': 0}
            else:
                nome = ''
                corAlert = {'b': 0, 'g': 0, 'r': 255}
            result = (1 - distanciaMinima) * 100
            cv2.rectangle(frame, (e, t), (d, b), (corAlert['b'], corAlert['g'], corAlert['r']), 2)
            texto = "{}".format(nome)
            #print("{} é aqui".format(b+200))
            if result:
                print('{} % de certeza'.format(round(result,2)))
            cv2.putText(frame, texto, (d,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Reconhecimento de Face', frame)
    cap.release()
    cv2.destroyAllWindows()
reconhecimentoFacial()