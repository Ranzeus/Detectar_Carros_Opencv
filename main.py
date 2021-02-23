import cv2
import numpy as np
import veiculo
import time

# criando um subtrator
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# capturando video
video = cv2.VideoCapture("surveillance.m4v")

# PEGANDO AS DIMENÇÕES DO VIDEO
largura = video.get(3)
altura = video.get(4)

#AREA DO FRAME
area_frame = altura*largura

#AREA QUE O CARRO OCUPA NO FRAME
area_carro_min = area_frame/400
area_carro_max = area_frame/50


print("area_carro_min: ",area_carro_min)
print("area_carro_max: ",area_carro_max)

###########################################
# LINHAS
line_up=int(2*(altura/5))
line_down=int(3*(altura/5))
############################################

# print("line_down:",line_up)
# print("line_down:",line_down)

######################################################
up_limit=int(1*(altura/5)) #LIMITE ACIMA
down_limit=int(4*(altura/5)) #LIMITE INFERIOR
######################################################

# CONTADORES DE VEICULOS SUBINDO E DESCENDO
cnt_up=0
cnt_down=0

#LIMITE INFERIOR
limit_infe=int(4*(altura/5))

#COR DA LINHA INFERIOR
cor_linha_infe=(255,0,255)

#COORDENADAS DE PONTOS DENTRO DA IMAGEM PARA DESENHAR A LINHA
pt1 =  [0, limit_infe]
pt2 =  [largura, limit_infe]

################################################################
#LINHA USANDO AS COORDENADAS DOS PONTOS
pts_L1 = np.array([pt1,pt2], np.int32)
#Adiciona um espano entra a primeira e a ultima linha da matriz
pts_L1 = pts_L1.reshape((-1,1,2))
################################################################

#MATRIZ DE 1s DE TAMANHO 3X3(ABRIR)
kernelzinho = np.ones((3,3),np.uint8)
#MATRIZ DE 1s DE TAMANHO 17X17(FECHAR)
kernelzao = np.ones((17,17),np.uint8)

################################
font = cv2.FONT_HERSHEY_SIMPLEX
################################

# array vazio de classes/objetos carros
cars = []

#variaveis para o caso de não detectar nada
max_p_age = 5
pid = 1

while(video.isOpened()):

    # ABRINDO VIDEO
    ret, frame = video.read()

    # testando se o frame não está vazio
    if frame is None:
        break

    # BORRANDO A IMAGEM COM GAUSIAN BLUR
    # blur = cv2.GaussianBlur(frame,(7,7),0)

    # PERCORRENDO ARRAY DE CARROS
    for i in cars:
        i.age_one()

    """
    Cada quadro é usado tanto para calcular a máscara do primeiro plano quanto para atualizar o plano de fundo. Se você 
    deseja alterar a taxa de aprendizagem usada para atualizar o modelo de fundo, é possível definir uma taxa de aprendizagem 
    específica passando um parâmetro para o método de aplicação.
    """
    fgMask = backSub.apply(frame)#aplicando a subtração de imagem

    # BINARIZAÇÃO
    ret, imBin = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    # APLICANDO KERNELZINHO (dilatação/ABRIR)
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelzinho)

    # APLICANDO KERNELZÃO (erosão/FECHAR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelzao)

    # ENCONTRANDO OS CONTORNOS
    contornos, hierarquia = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # DESENHAR O QUADRADO EM TORNO DE CADA CARRO
    for contorno in contornos:
        area = cv2.contourArea(contorno)

        #Verificação do tamnho da área para eliminar os lixos indentificados
        if area > area_carro_min and area < area_carro_max:

            #MEDIA PONDERADA DA INTENSIDADE DOS PIXELS, USADA PARA ENCONTRAR O CENTRO DO CONTORNO
            m = cv2.moments(contorno)

            # ENCONTRANDO AS COORDENADAS DO CENTRO DO CONTORNO
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])

            (x,y,l,a) = cv2.boundingRect(contorno)
            # print("x=",x,"y=",y,"l=",l,"a=",a)

            new = True
            if cy in range(up_limit, down_limit):
                for i in cars:

                    if abs(x - i.getX()) <=l and abs(y - i.getY()) <= a:
                        new = False
                        i.updateCoords(cx, cy)

                        # print("UP: ", i.going_UP(line_down,line_up))
                        # print("DOWN: ",i.going_DOWN(line_down,line_up))

                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1

                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1

                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        index = cars.index(i)
                        cars.pop(index)
                        del i

                if new == True:  # If nothing is detected,create new
                    p = veiculo.Car(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

                #DESENHANDO O CIRCULO NAS COODENADAS DO CENTRO DA IMAGEM
                #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                #DESENHANDO RETANGULO
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 1)

    #MONTANDO STRING PARA IMPRIMIR
    str_down = 'DOWN: ' + str(cnt_down)
    str_up = 'UP: ' + str(cnt_up)

    #IMPRIMINDO O CONTADOR
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
