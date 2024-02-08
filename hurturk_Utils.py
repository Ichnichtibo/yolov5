import numpy as np



def etraftaki_indeksleri_bul(matris, satir, sutun, mesafe=1):
    satir_sayisi = len(matris)
    sutun_sayisi = len(matris[0])
    
    indeksler = []
    
    for i in range(satir - mesafe, satir + mesafe + 1):
        for j in range(sutun - mesafe, sutun + mesafe + 1):
            if 0 <= i < satir_sayisi and 0 <= j < sutun_sayisi:
                indeksler.append((i, j))
    
    return indeksler


def alanBol(x,xSayi,ySayi):
    # print(np.shape(x))
    height,width,a = x.shape
    xShape,yShape = int(width/(xSayi)),int(height/(ySayi))
    # print(xShape,yShape)
    xCord = 0   
    yCord = 0
    tempList = []
    Cords = []
    while True:
        tempList.append([(xCord,yCord),(xCord+xShape,yCord+yShape)])
        xCord += int(xShape/3)
        if xCord+xShape >width:
            yCord+=int(yShape/3)
            xCord = 0
            Cords.append(tempList)
            tempList = []
        if yCord+yShape >= height:
            print(Cords)
            return Cords