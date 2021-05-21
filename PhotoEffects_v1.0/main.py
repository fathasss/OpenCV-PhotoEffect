#%% Library
import cv2
import numpy as np

#%% Photo Import
image_path = "my_db/foto.png"
img = cv2.imread(image_path)

#%% Functions
def colorEffect(image,k):
    data = np.float32(image).reshape(-1,3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret,label,center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result

#%% Data
total_color = 7
img = colorEffect(img,total_color)

#%% Application Run
cv2.imshow("Resim",img)
cv2.waitKey(0)
cv2.destroyAllWindows ()
# %%
