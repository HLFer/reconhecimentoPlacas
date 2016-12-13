"""TP2 de PDI."""

# Bibliotecas Utilizadas
import cv2  # OpenCV
import numpy as np  # Numpy
import os
import sys


def validar(cont):
    """Valida contornos."""
    rect = cv2.minAreaRect(cont)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = rect[1][0]
    height = rect[1][1]
    if ((width != 0) and (height != 0)):
        razao1 = height / width
        razao2 = width / height
        mult = height * width
        if (((razao1 > 3) and (razao1 < 10) and (height > width)) or ((razao2 > 3) and (razao2 < 10) and (width > height))):
            if((mult < 20000) and (mult > 3000)):
                return True
    return False


def encontra_placas(caminho_imagem):
    """Processa imagem para encontrar placas."""
    original = cv2.imread(caminho_imagem)
    cv2.imshow(caminho_imagem, original)

    # converter para escala de cinza
    original_cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # suaviza a imagem
    suavizada = cv2.GaussianBlur(original_cinza, (3, 3), 0)
    cv2.imwrite(diretorio + "1 - suavizada.jpg", suavizada)

    # encontra as bordas da imagem com operador Sobel
    sobel = cv2.Sobel(suavizada, cv2.CV_8U, 1, 0, ksize=3)
    cv2.imwrite(diretorio + "2 - sobel.jpg", sobel)

    # # ETAPA 1 : TopHat
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    # tophat = cv2.morphologyEx(suavizada, cv2.MORPH_TOPHAT, kernel)
    # cv2.imwrite(diretorio + "3 - tophat.jpg", tophat)

    # ETAPA 2 : Binarizacao
    binarizada = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(diretorio + "4 - binarizada.jpg", binarizada)

    # ETAPA 3 : Fechamento HOriz
    kernel = np.ones((1, 60), np.uint8)
    fechamento = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(diretorio + "5 - fechamento.jpg", fechamento)

    #  ETAPA 4 : Abertura max e min (Vertical)
    kernel = np.ones((6, 1), np.uint8)
    abertura = cv2.morphologyEx(fechamento, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(diretorio + "7 - abertura.jpg", abertura)

    #  ETAPA 4 : Abertura max e min (Horizontal e Vertical)
    # kernel = np.ones((25, 1), np.uint8)
    # abertura2 = cv2.morphologyEx(abertura, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite(diretorio + "8 - abertura2.jpg", abertura2)

    #  ETAPA 5 : Dilatacao e remocao de ruidos
    # kernel = np.ones((1, 55), np.uint8)
    # fechamento3 = cv2.morphologyEx(abertura2, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(diretorio + "10 - fechamento3.jpg", fechamento3)

    # ETAPA 6 : Dilatação.
    kernel = np.ones((5, 5), np.uint8)
    dilatacao_final = cv2.dilate(abertura, kernel, iterations=1)
    cv2.imwrite(diretorio + "11 - dilatacao_final.jpg", dilatacao_final)

    # Encontra contorno das áreas segmentadas após abertura.
    contours = cv2.findContours(dilatacao_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    placas = []

    # Cria uma caixa retangular em torno dos contornos encontrados validando-os e armazenando em uma lista.
    for cont in contours:
        # cv2.drawContours(original, [box], 0, (0, 255, 0), 2)
        if validar(cont):
            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            center = (int(rect[0][0]), int(rect[0][1]))
            placas.append([width, height, center])
            # cv2.drawContours(original, [box], 0, (0, 0, 255), 2)

    # cv2.imwrite(diretorio + "caixa_original.jpg", original)

    # Lista com imagens das placas.
    placa_img = []

    if(len(placas) >= 1):
        print(caminho_imagem + ":\n\t Placas encontradas\n")
        if (placas[0][0] < placas[0][1]):
            # Recorta placa da imagem.
            placa_img = cv2.getRectSubPix(original, (placas[0][1] + 15, placas[0][0] + 5), placas[0][2])
        else:
            # Recorta placa da imagem.
            placa_img = cv2.getRectSubPix(original, (placas[0][0] + 5, placas[0][1] + 15), placas[0][2])
        cv2.imshow("Placa", placa_img)
        cv2.imwrite(diretorio + "12 - placa_segmentada.jpg", placa_img)

    else:
        # Se a lista estiver vazia, nenhuma placa foi encontrada.
        print(caminho_imagem + ":\n\t Nenhuma placa encontrada\n")

    # Espera usuário apertar uma tecla para sair.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


diretorio = "resultado/"
rootdir = "Treinamento-placas/"

# Limpa pasta de resultados.
for the_file in os.listdir(diretorio):
        file_path = os.path.join(diretorio, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

if len(sys.argv) < 2:
    # Percorre imagens da pasta de treinamento.
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file[1] != '_':
                nome_do_arquivo = os.path.join(rootdir, file)
                encontra_placas(nome_do_arquivo)
else:
    # analisa somente uma imagem.
    encontra_placas(str(sys.argv[1]))
