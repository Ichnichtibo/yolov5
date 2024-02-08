import torch
import cv2
import numpy as np
import hurturk_Utils
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

class HurturkTakip:
    teyit1:bool = False
    constI:int = 0 
    constJ:int = 0
    roiCord = []
    sayac= 0

    def __init__(self,camIndex = 0):
        # YOLOv5 modelini yükle
        self.device = select_device('')
        self.model = attempt_load('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        # Kamera ayarları
        self.cap = cv2.VideoCapture(camIndex=0)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cords = hurturk_Utils.alanBol(self.cap.read()[,1],3,3)
        self.rowNum,self.coluomNum = np.shape(self.cords)[0:2]
        print(self.rowNum,self.coluomNum)

    def ilkYakalama(self,roi,i,j):
        results = self.model(roi)  # 640'lık giriş boyutu kullan
        # Non-maximum suppression ile sonuçları filtrele
        results = non_max_suppression(results, 0.4, 0.5)
   
        for _, det in enumerate(results):
            det[:, :4] = det[:, :4].clip(0, roi.shape[0])  # Sınırları görüntüye uygun şekilde sınırla
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self.model.names[c]} {conf:.2f}'
                cv2.rectangle(roi, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 1)

        if len(results)>0:
            self.constJ = j
            self.constI = i 
            self.teyit1 = True
    

    def roideArama(self,frame):
        for i,j in self.roiCord:
            roi = frame[self.cords[i][j][0][1]:self.cords[i][j][1][1],
                        self.cords[i][j][0][0]:self.cords[i][j][1][0]]
            results = self.model(roi)  # 640'lık giriş boyutu kullan
            # Non-maximum suppression ile sonuçları filtrele
            results = non_max_suppression(results, 0.4, 0.5)
            
            if len(results)>0:
                self.constI = i
                self.constJ = j
                break
    
    def ortaKaredenCiktimi(self,frame):
        roi = frame[self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][1]+109:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][1]-109,
                    self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][0]+60:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][0]-60]
        cv2.imshow("b",roi)

        results = self.model(roi)  # 640'lık giriş boyutu kullan
        # Non-maximum suppression ile sonuçları filtrele
        results = non_max_suppression(results, 0.4, 0.5)

        if len(results) == 0:
            print("ekranı değiştir")
            self.roideArama(frame=frame)
            
    def ikinciYakalama(self,frame,kareSayisi):
        self.roiCord = hurturk_Utils.etraftaki_indeksleri_bul(self.cords,self.constI,self.constJ,kareSayisi) #!
        print(self.roiCord)

        roi = frame[self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][1]:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][1],
                    self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][0]:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][0]]
        
        self.ortaKaredenCiktimi(frame=frame)
            

        cv2.imshow("constRoi",roi)

        results = self.model(roi)  # 640'lık giriş boyutu kullan
        # Non-maximum suppression ile sonuçları filtrele
        results = non_max_suppression(results, 0.4, 0.5)
    
        for _, det in enumerate(results):
            det[:, :4] = det[:, :4].clip(0, roi.shape[0])  # Sınırları görüntüye uygun şekilde sınırla
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self.model.names[c]} {conf:.2f}'
                cv2.rectangle(roi, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 1)

    def __call__(self):
        i = j = 0
        
        self.sayac = 240
        while True:
            self.teyit1 = False 
            ret, frame = self.cap.read()
            frame = cv2.flip(frame ,1)

            j+=1
            if j >= self.coluomNum:
                i +=1
                j =0
            if i >= self.rowNum:
                i = 0 

            #cords[i][j][0][1] -> ilk noktanın y
            #cords[i][j][0][0] -> ilk noktanın x
            #cords[i][j][1][1] -> ikinci noktanın y
            #cords[i][j][1][0] -> ikinci noktanın x
                
                
            cv2.rectangle(frame,self.cords[i][j][0],self.cords[i][j][1],(0,0,255),1)
            cv2.rectangle(frame,self.cords[self.rowNum-i-1][self.coluomNum-j-1][0],self.cords[self.rowNum-i-1][self.coluomNum-j-1][1],(0,0,255),1)
            self.ilkYakalama(roi=frame[self.cords[i][j][0][1]:self.cords[i][j][1][1],
                                       self.cords[i][j][0][0]:self.cords[i][j][1][0]],
                                       i=i,j=j)
            self.ilkYakalama(roi=frame[self.cords[self.rowNum-i-1][self.coluomNum-j-1][0][1]:self.cords[self.rowNum-i-1][self.coluomNum-j-1][1][1],
                                       self.cords[self.rowNum-i-1][self.coluomNum-j-1][0][0]:self.cords[self.rowNum-i-1][self.coluomNum-j-1][1][0]],
                                       i=self.rowNum-i-1,j=self.coluomNum-j-1)
            
          
            if self.teyit1 and self.sayac >= 200: 
                  
                self.teyit2 = False
                self.sayac = 0
            if self.sayac <= 200:
                # print("Takip çalıştırıldı")
                self.ikinciYakalama(frame,1)
                self.sayac += 1.4
            if ret:
                cv2.imshow("a",frame)            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("silindi")

a = HurturkTakip()
a()

