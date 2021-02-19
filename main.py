import cv2
import numpy as np

# criando um subtrator
backSub = cv2.createBackgroundSubtractorMOG2()

# capturando video
video = cv2.VideoCapture("Freewa.mp4")

# PEGANDO AS DIMENÇÕES DO VIDEO
altura = video.get(3)
largura = video.get(4)

#VARIAVEIS PARA DESENHAR OS QUADRADOS
min_x,min_y = largura, altura
max_x = max_y = 0

kernelzinho = np.ones((3,3),np.uint8) #MATRIZ DE 1s DE TAMANHO 3X3
kernelzao = np.ones((17,17),np.uint8) #MATRIZ DE 1s DE TAMANHO 17X17

while(video.isOpened()):

    # ABRINDO VIDEO
    ret, frame = video.read()

    # testando se o frame não está vazio
    if frame is None:
        break

    blur = cv2.GaussianBlur(frame,(7,7),0)

    """
    Cada quadro é usado tanto para calcular a máscara do primeiro plano quanto para atualizar o plano de fundo. Se você 
    deseja alterar a taxa de aprendizagem usada para atualizar o modelo de fundo, é possível definir uma taxa de aprendizagem 
    específica passando um parâmetro para o método de aplicação.
    """
    fgMask = backSub.apply(blur)#aplicando a subtração de imagem

    # BINARIZAÇÃO
    ret, imBin = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    # APLICANDO KERNELZINHO (dilatar/ABRIR)
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelzinho)

    # APLICANDO KERNELZÃO (corroer/FECHAR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelzao)

    # ENCONTRANDO OS CONTORNOS
    contornos, hierarquia = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # DESENHAR O QUADRADO EM TORNO DE CADA CARRO
    for contorno in contornos:
        (x,y,l,a) = cv2.boundingRect(contorno)
        # print("x=",x,"y=",y,"l=",l,"a=",a)
        min_x, max_x = min(x, min_x), max(x + l, max_x)
        min_y, max_y = min(y, min_y), max(y + a, max_y)
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 1)

    # EXPOR OS FRAMES
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
