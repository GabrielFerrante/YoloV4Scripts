import cv2

#lendo uma imagem
imagem = cv2.imread('pessoas.jpg')

#usando o arquivo haarcascade xml
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#converter a imagem para cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#vetor de deteccoes
#MultiScale parametros:
"""
Scale Factor = redimensiona a imagem de acordo com uma escala, maior para menor
quanto menor o parametro, mais lento fica a performance

Min Neighbors = quantidade minima para cada retângulo candidato
deve ter para mantê-lo. Maiores valores + Precisão, Valores Menores + detecções

Min Size = Especifica o menor objeto a ser reconhecido
valor padrão, 30 x 30

Max Size = Especifica o maior objeto a ser reconhecido
"""
deteccoes = detector.detectMultiScale(imagem_cinza, scaleFactor=1.09, 
                                        minNeighbors=5) #minSize=(30,30), 
                                        #maxSize=(40,40))

#Quantas detecções
print(deteccoes)

#colando o bounding box

"""
X = eixo X
Y = eixo Y
w = Width
h = Height
"""
for (x, y, w, h) in deteccoes:
    #recebe a imagem original, os pontos iniciais (x e y), os pontos finais somando os iniciais com
    #a altura e largura, uma cor em BGR, grossura da linha 
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,255,0), 2)


#Apresentando o detector
cv2.imshow('Detector de faces', imagem)
#Fecha a janela apertando qualquer tecla
cv2.waitKey(0)
#Liberar espaço na memória
cv2.destroyAllWindows()