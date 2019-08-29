import serial
import matplotlib.pyplot as plt
from drawnow import *
import atexit
import time
 
values = []
plt.ion()#�򿪽���ģʽ
 
 
serialArduino = serial.Serial('COM14', 9600)
 
def plotValues():
    plt.title('Serial temperature from Arduino')
    plt.grid(True)
    plt.ylabel('temperature')
    plt.plot(values,'rx-', label='temperature')
    plt.legend(loc='upper right')
 
def doAtExit():
    serialArduino.close()
    print("Close serial")
    print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))
 
atexit.register(doAtExit)#�����˳�ʱ���ص�����
 
print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))
 
#Ԥ������������
for i in range(0,50): 
    values.append(0)
     
while True:
    while (serialArduino.inWaiting()==0):
        pass
    print("readline()")
    valueRead = serialArduino.readline(500)
 
    #����Ƿ����������Чֵ
    try:
        valueInInt = float(valueRead)
        print(valueInInt)
        if valueInInt <= 1024:
            if valueInInt >= 0:
                values.append(valueInInt)
                values.pop(0)
                drawnow(plotValues)
            else:
                print("Invalid! negative number")#��Ч  ����
        else:
            print("Invalid! too large")# ��Ч ����1024
    except ValueError:
        print("Invalid! cannot cast")