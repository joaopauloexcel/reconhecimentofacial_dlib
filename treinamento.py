import cv2
import time
import os
import glob #serve para percorrer uma pasta
import _pickle as cPickle
import dlib
import numpy as np

def treino(senha):
        detectorFace = dlib.get_frontal_face_detector()  # detector de faces
        detectorPontos = dlib.shape_predictor(
            "recursos/shape_predictor_68_face_landmarks.dat")  # detector de pontos faciais
        # rede neural convulacional
        reconhecimentoFacial = dlib.face_recognition_model_v1(
            "recursos/dlib_face_recognition_resnet_model_v1.dat")  # variavel para o reconhecimento facial

        descritoresFaciais = None
        indice = {}
        idx = 0
        for arquivo in glob.glob(
           os.path.join("fotos/treinamento", "*.jpg")):  # pega o caminho onde está salvo o arquivo
           imagem = cv2.imread(arquivo)
           # aqui detectamos todas as faces
           facesDetectadas = detectorFace(imagem, 1)  # detecta as faces
           numeroFacesDetectadas = len(facesDetectadas)  # numero de faces detectadas
           if numeroFacesDetectadas > 1:  # dlib se perde com mais de uma face na imagem para treinamento
               print("Há mais de uma face na imagem {}".format(arquivo))
               exit(0)
           elif numeroFacesDetectadas < 1:  # se nenhuma face estiver no arquivo de treinamento
               print("Nenhuma face encontrada no arquivo {}".format(arquivo))
               exit(0)
                                # aqui percorremos cada face para pegar os pontos faciais
           for face in facesDetectadas:  # percorrer a matriz de faces
              # aqui pegamos os pontos faciais
               pontosFaciais = detectorPontos(imagem, face)  # detecta os pontos faciais
              # aqui pegamos o array com 128 pontos da face
               descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
              # Aqui geramos uma lista dos descritores faciais
               listaDescritorFacial = [df for df in descritorFacial]
              # aqui convertemos ele para o formato do numpy que precisaremos para gerar os arquivos
               npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
              # até aqui a dimensão do array é 128, precisamos que tenha 1x128, para isso, acrescentamos uma nova coluna
               npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
               print(npArrayDescritorFacial)
               if descritoresFaciais is None:
                    descritoresFaciais = npArrayDescritorFacial
               else:
                    descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial),
                                                                            axis=0)
               indice[idx] = arquivo
               idx += 1
        np.save("recursos/descritores_{}.npy".format(senha), descritoresFaciais)
        with open("recursos/indices_{}.pickle".format(senha), 'wb') as f:
           cPickle.dump(indice, f)