from zmqRemoteApi import RemoteAPIClient
import numpy as np
import cv2

# start connection to coppelia
client = RemoteAPIClient()
sim = client.getObject('sim')

# acquire objects
cam = sim.getObject('/Cam')

client.setStepping(True)
sim.startSimulation()

img, resX, resY = sim.getVisionSensorCharImage(cam)
img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
cv2.imwrite('test.jpg', img)

sim.stopSimulation()
