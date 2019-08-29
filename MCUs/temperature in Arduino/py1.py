import serial
import matplotlib.pyplot as plt
from drawnow import *
import atexit
import time
 
values = []
plt.ion()#打开交互模式
 
 
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
 
atexit.register(doAtExit)#程序退出时，回调函数
 
print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))
 
#预加载虚拟数据
for i in range(0,50): 
    values.append(0)
     
while True:
    while (serialArduino.inWaiting()==0):
        pass
    print("readline()")
    valueRead = serialArduino.readline(500)
 
    #检查是否可以输入有效值
    try:
        valueInInt = float(valueRead)
        print(valueInInt)
        if valueInInt <= 1024:
            if valueInInt >= 0:
                values.append(valueInInt)
                values.pop(0)
                drawnow(plotValues)
            else:
                print("Invalid! negative number")#无效  负数
        else:
            print("Invalid! too large")# 无效 超过1024
    except ValueError:
        print("Invalid! cannot cast")