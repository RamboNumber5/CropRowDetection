from unetRGB import *
from tensorflow.keras.preprocessing.image import array_to_img


print("array to image")
imgs = np.load('./results/imgs_mask_test.npy')
piclist = []
for line in open("./results/pic.txt"):
    line = line.strip()
    picname = line.split('/')[-1]
    piclist.append(picname)
for i in range(imgs.shape[0]):
    path = "./results/" + piclist[i]
    img = imgs[i]
    img = array_to_img(img)
    img.save(path)
    cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv_pic = cv2.resize(cv_pic, (512, 512), interpolation=cv2.INTER_CUBIC)
    binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(path, cv_save)



