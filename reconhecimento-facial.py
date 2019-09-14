import os
import dlib
import numpy as np
import cv2

# Método que será acionado para realizar o reconhecimento facial
def reconhecimentoFacial(identify):
    cap = cv2.VideoCapture(0)# habilita webcam

    # detector de faces frontal do próprio dlib
    detectorFace = dlib.get_frontal_face_detector()
    # detector de pontos faciais do próprio dlib
    detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
    reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

    try: # Cria um vetor com os indices.@@
        indices = np.load("recursos/indices_{}.pickle".format(identify), allow_pickle=True)
        # Carrega as caracteristicas extraidas das faces treinadas.@@
        descritoresFaciais = np.load("recursos/descritores_{}.npy".format(identify), allow_pickle=True)
        # se deu certo abrir os arquivos, instancio variável de margem de acerto de 0 a 1 para o reconhecimento da face
        limiar = 0.5 # com até 50% de erro, quero ainda dizer que a pessoa foi reconhecida
        count=0
    except: #trata erro de abertura dos arquivos
        print('Rede não treinada')
        exit(0)
    while cap.isOpened(): #enquanto câmera estiver aberta
        conectado, video = cap.read() # realiza a leitura da imagem da câmera
        #video = cv2.resize(video, (400,300))
        frame = cv2.flip(video, 2)#padroniza espelhamento da webcam, pois cada uma pode espelhar a imagem de um jeito.
        frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#passa cada frame obtido para a escala de cinza
        if(count % 2 == 0):
            facesDetectadas = detectorFace(frameCinza)# Guarda o array de faces detectadas na escala de cinza
            if facesDetectadas: # se houver faces detectadas
                for face in facesDetectadas: # para cada face detectada
                    #pega 4 pontos extemos de cada uma delas, esquerda, top, direita, baixo
                    e, t, d, b = (int(face.left()), int(face.top()),
                                  int(face.right()), int(face.bottom()))
                    # pontosFaciais recebem o detector de pontos de cada face na escala de cinza
                    pontosFaciais = detectorPontos(frameCinza, face)
                    # O descritorFacial recebe a função que retorna o resultado de comparação
                    # entre os frames transmitidos na webcan com os pontos faciais obtidos na imagem
                    # aqui não tem como ser escala de cinza, pois essa função trabalha com escala RGB
                    descritorFacial = reconhecimentoFacial.compute_face_descriptor(frame, pontosFaciais)
                    # aqui é criado uma lista contendo os resultados obtidos da comparação acima
                    listaDescritorFacial = [fd for fd in descritorFacial]
                    # aqui convertemos essa lista para o formato do numpy para ser
                    # comparada com os arquivos na pasta "recursos"
                    npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
                    # até aqui a dimensão de cada array é 128, precisamos que tenha 1x128,
                    # onde dizemos que a 1 face pertence 128 caracteristicas. Para isso, acrescentamos uma nova coluna
                    npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
                    # calculo da distancia de diferenciação das características obtidas na webcam
                    # com as características salvas no arquivo de treinamento "descritores.npy" na pasta "recursos".
                    distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
                    # Aqui é retornado os índices dos valores mínimos obtidos pela 'distancias' ao longo das comparações
                    minimo = np.argmin(distancias)
                    # aqui, a variável "distanciaMinima" recebe 'distancias' na posição do menor valor de diferenciação encontrado
                    distanciaMinima = distancias[minimo]
                    # Aqui, é feito a comparação se o valor mínimo encontrado nos descitores é menor ou igual a minha margem de erro
                    if distanciaMinima <= limiar:
                        # se for, busco no arquivo de indice a parte salva depois do _,
                        # onde coloquei a informação do nome da pessoa da foto tirada
                        nome = os.path.split(indices[minimo])[1].split("_")[0]
                        # Aqui, estou colorindo a variável que será utilizada no retânculo de "Verde"
                        corAlert = {'b': 0, 'g': 255, 'r': 0}
                    else: # caso a distância de diferenciação seja maior que a margem de erro que estipulei,
                        nome = ''
                        # Aqui, estou colorindo a variável que será utilizada no retânculo de "Vermelho"
                        corAlert = {'b': 0, 'g': 0, 'r': 255}
                    # Aqui obtemos o resultado de acerto de cada reconhecimento, lembrando que o limiar
                    #aceitará uma face como reconhecida se essa tiver de 50% de acerto para cima
                    result = (1 - distanciaMinima) * 100
                    # Aqui desenho o retângulo na tela com a cor desejada e na posição dos pontos da face.
                    cv2.rectangle(frame, (e, t), (d, b), (corAlert['b'], corAlert['g'], corAlert['r']), 2)
                    # Aqui armazeno o nome da pessoa reconhecida, se ela não for reconhecida, aqui chegará ''
                    texto = "{}".format(nome)
                    # Se houver resultado obtido de margem de acerto
                    if result:
                        # Aqui, estou mandando printar no console a porcentagem de acerto do reconhecimento de uma pessoa
                        print('{} % de certeza'.format(round(result,2)))
                    # Aqui, estou mandando esceve na tela o nome da pessoa no topo direito do retângulo.
                    cv2.putText(frame, texto, (d,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        # pressionou 'q', sai da aplicação.
        if cv2.waitKey(1) == ord('q'):
            break
        # Abre a imagem da webcan na tela.
        cv2.imshow('Reconhecimento de Face', frame)
        print(count)
        count+=1
    cap.release()
    cv2.destroyAllWindows()

# Chamando  o método aqui
reconhecimentoFacial('123')