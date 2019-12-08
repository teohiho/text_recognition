import numpy as np
import timeit
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import struct       #modun dung de dinh dạng ban ghi nhi phan , giai nen du lieu #https://www.geeksforgeeks.org/struct-module-python/
import pickle
from skimage import io  # pip install scikit-image
import cv2
from PIL import ImageEnhance, Image

# TRAIN_ITEMS = 112799
# TEST_ITEMS = 18799

TRAIN_ITEMS = 1300
TEST_ITEMS = 1300

# train-images-idx3-ubyte: đào tạo tập hình ảnh
# đào tạo-nhãn-idx1-ubyte: nhãn tập huấn luyện
# t10k-images-idx3-ubyte: kiểm tra tập hình ảnh
# t10k-labels-idx1-ubyte: nhãn thiết lập thử
#Tập huấn luyện có 60000, bài kiểm tra 10000

######################################## Rotate 90 ################################################
HEIGHT = 28
WIDTH =28
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


######################################## Read data MNIST ##########################################
def loadMnistData():
    mnist_data = []
    for img_file,label_file,items in zip(['data/train-images-idx3-ubyte','data/t10k-images-idx3-ubyte'],
                                   ['data/train-labels-idx1-ubyte','data/t10k-labels-idx1-ubyte'],
                                   [TRAIN_ITEMS, TEST_ITEMS]):
    # ===
    # for img_file,label_file,items in zip(['../dataset/notMNIST-to-MNIST/train-images-idx3-ubyte','../dataset/notMNIST-to-MNIST/t10k-images-idx3-ubyte'],
    #                                ['../dataset/notMNIST-to-MNIST/train-labels-idx1-ubyte','../dataset/notMNIST-to-MNIST/t10k-labels-idx1-ubyte'],
    #                                [TRAIN_ITEMS, TEST_ITEMS]):
    # ===
    # for img_file,label_file,items in zip(['data/emnist-letters-train-images-idx3-ubyte','data/emnist-letters-test-images-idx3-ubyte'],
    #                                ['data/emnist-letters-train-labels-idx1-ubyte','data/emnist-letters-test-labels-idx1-ubyte'],
    #                                [TRAIN_ITEMS, TEST_ITEMS]):
        data_img = open(img_file, 'rb').read()
        data_label = open(label_file, 'rb').read()

        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset)
       
        offset += struct.calcsize(fmt)
      
        image_size = height * width
      
        fmt = '>{}B'.format(image_size)
     
        if items > img_number:
            items = img_number
        images = np.empty((items, image_size))
        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
           
            images[i] = images[i]/256
            # images[i] = images[i]
            offset += struct.calcsize(fmt)


        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)
        # print('magic number is {} and label number is {}'.format(magic_number, label_number))
       
        offset += struct.calcsize(fmt)
        #B means unsigned char
        fmt = '>B'
       
        if items > label_number:
            items = label_number
        labels = np.empty(items)
        for i in range(items):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)
        
        mnist_data.append((images, labels.astype(int)))
    return mnist_data

######################################## Train data ##########################################
def train_model_SVM():
    print("Bắt đầu train!")
    start_time = timeit.default_timer()
    print("+ Start_time: ", start_time)

    training_data, test_data = loadMnistData()
    # print("training_data: ", training_data)
    # print("test_data: ", test_data)

    # ######### Rotate 90 độ #############
    # for i in range(len(training_data[0]-1)):
    #     logo = training_data[0][i].reshape(28, 28)
    #     logo = rotate(logo)
    #     training_data[0][i] = logo.reshape(784,)
    # for i in range(len(test_data[0]-1)):
    #     logo = test_data[0][i].reshape(28, 28)
    #     logo = rotate(logo)
    #     test_data[0][i] = logo.reshape(784,)

    # train
    # classifier = svm.SVC()        #9443 trong  10000 gía trị đúng.
    classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False) #9824 trong 10000 gía trị đúng.

    # cho nó học từ images và label của data train
    classifier.fit(training_data[0], training_data[1])         
    train_time = timeit.default_timer()
    print("+ Train_time: ", train_time)      
    print('+ Gemfield train cost {}'.format(str(train_time - start_time) ) )

    pickle.dump(classifier, open("AlphabetModel_FromHienDataset_SVM", 'wb'))

    # test
    print("Bắt đầu test!")
    set_score("AlphabetModel_FromHienDataset_SVM")
    set_predictions("AlphabetModel_FromHienDataset_SVM")

    test_time = timeit.default_timer()
    print('+ Gemfield test cost {}'.format(str(test_time - train_time) ) )          #gemfield test cost 206.6903916629999


######################################## Train data Nertron Network ##########################################
def train_model_NN():
    print("Bắt đầu train!")
    start_time = timeit.default_timer()
    print("+ Start_time: ", start_time)

    training_data, test_data = loadMnistData()
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), 
                    max_iter=480, alpha=1e-4,
                    solver='sgd', verbose=10, 
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1)

    mlp.fit(training_data[0], training_data[1])

    train_time = timeit.default_timer()
    print("+ Train_time: ", train_time)
    print('+ Gemfield train cost {}'.format(str(train_time - start_time) ) )

    pickle.dump(mlp, open("AlphabetModel_FromHienDataset_NN", 'wb'))

    # test
    print("Bắt đầu test!")
    set_score("AlphabetModel_FromHienDataset_NN")
    set_predictions("AlphabetModel_FromHienDataset_NN")

    test_time = timeit.default_timer()
    print('+ Gemfield test cost {}'.format(str(test_time - train_time) ) )


##################################### Xem thử % độ chính xác của Model vừa train ##############################
def set_score(nameModel):
    classifier = pickle.load(open(nameModel, 'rb'))
    training_data, test_data = loadMnistData()

    print("Training set score: %f" % classifier.score(training_data[0], training_data[1]))
    print("Test set score: %f" % classifier.score(test_data[0], test_data[1]))

def set_predictions(nameModel):
    classifier = pickle.load(open(nameModel, 'rb'))
    training_data, test_data = loadMnistData()

    #cho ra các label của test gọp lại thành mảng
    predictions = classifier.predict(test_data[0])
    predictions = []
    for a in classifier.predict(test_data[0]):
        predictions.append(a)

    # print("PREDICT %r" % predictions)

    # so sánh cái mảng các label vừa được dự đoán được với mảng label mà ban đầu đã cho để xem có đúng thay hông??
    i = 0
    for a, y in zip(predictions, test_data[1]):
        if a == y:
            i = i + 1
    num_correct = i
    # print("predictions", predictions)  # [7,2,1,..]
    print("%s trong %s gía trị đúng." % (num_correct, len(test_data[1])))      


##################################### Test image ##############################
def predict_image(img):
    logo = img
    if type(img) is str:
        logo = io.imread(img)
    classifier = pickle.load(open("handwrite_model_EMNIST", 'rb'))
    show_image(logo)
    logo = logo.reshape(1, -1)
    result = classifier.predict(logo)
    print("RESULT %r" % result)
    return result


def show_image(img):
    logo = img.reshape(28, 28)
    print(logo.shape)
    print(len(logo[0]))
    for i in range(logo.shape[0]):
        for j in range(logo.shape[1]):
            if logo[i][j] > 0.0:
                print("@", end="")
            else:
                print("-", end="")
        print()


def image_predict_image(img):
    img = cv2.imread(img,0)
    img = cv2.resize(img,(28, 28)).astype(np.float32)
    print("img:",img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)

    logo = img
    if type(img) is str:
        logo = io.imread(img, as_grey=True)
    classifier = pickle.load(open("handwrite_model_EMNIST", 'rb'))
    logo_train = (logo*256).reshape(1, -1)
    total_pixel = 28*28
    logo_train_chia = [[0 for _ in range(total_pixel)]]
    for i in range(total_pixel):
        logo_train_chia[0][i] = logo_train[0][i] / 256

    show_image(logo)

    result = classifier.predict(logo_train_chia)
    print("RESULT %r" % result)
    return result[0]

##################################### Convert Image Color to Gray resize #####################################


def image_color_to_gray_size(imageSimple, signal):
    # cv2.imshow("filename",filename )
    img = binarize_image(imageSimple, 200)
    
    img = cv2.resize(img,(28, 28))

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    
    logo = img

    if type(img) is str:
        logo = io.imread(img, as_grey=True)
    print("Load Model")
    classifier = pickle.load(open("AlphabetModel_FromHienDataset_NN", 'rb'))
    
    # logo = logo.reshape((logo.shape[0]*3, 28, 28))
    # logo = np.arange(2352).reshape(1,28, 28 )
    
    logo_train = (logo).reshape(1, -1)

    total_pixel = 28*28
    logo_train_chia = [[0 for _ in range(total_pixel)]]
    
    for i in range(total_pixel):
        logo_train_chia[0][i] = logo_train[0][i] / 256

    show_image(logo)
 
    result = classifier.predict(logo_train_chia)[0]

    print("The predicted letter is :")

    # Xử lý chữ i
    if(signal == 'no'):
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "rf", "rt"]
        print(alphabet[result- 1])
        writeListToTextFile(alphabet[result- 1],'result.txt', 'a')
    elif(signal == 'yes'):
        print("i")
        writeListToTextFile("i",'result.txt', 'a')
    elif(signal == 'double'):
        print("")
        writeListToTextFile("",'result.txt', 'a')
    

def binarize_image(imageSimple, threshold):
    """Binarize an image."""
    image_file = Image.fromarray(imageSimple)
    image_file = ImageEnhance.Contrast(image_file).enhance(3.5)
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    # print(">>>>>>>" + str(numpy_array[0][0][0]))
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j]> threshold:
                numpy_array[i][j]  = 0
            else:
                numpy_array[i][j]  = 255
    # img = cv2.resize(numpy_array,(28, 28))
    return numpy_array





def writeListToTextFile(list, filePath, mode='a'):
    ''' Write list to csv line by line '''
    with open(filePath, mode, encoding="utf8") as myfile:
        for item in list:
            # myfile.write(str(item) +  '\n')
            myfile.write(str(item) )



def signalToTheEndOfAWord(signal):
    if(signal == 'end'):
        writeListToTextFile(' ', 'result.txt', 'a')



# def show_image1(img):
#     logo = img
#     if type(img) is str:
#         logo = io.imread(img)

#     show_image(logo)







####-----------------------
#-------- De train du lieu----------------------
# train_model_SVM()
# train_model_NN()

#-------- Doc du lieu tu bo du lieu MNist--------
# training_data, test_data = loadMnistData()

#--------
# print(test_data[1][221:400])
# print(test_data[0][45])
# show_image(test_data[0][45])

#-------- TEST DATA -------------------
# training_data, test_data = loadMnistData()
# predict_image(rotate(test_data[0][3523]))

#-------- Chia anh cho 256 roi Doan anh la so nao------
# image_predict_image("/home/admin/teo/images/image_0.jpg")

#dung
# image_predict_image("/home/dell/Documents/TEO/ai-python/digit_prediction-master/temp.jpg")


#thu
# logo = io.imread("/home/teo/STUDY/digit_prediction/temp.jpg", as_grey=True)
# logo1 = misc.imresize(logo, (28,28))
# image_predict_image(logo1)
# print(logo.shape)       #28*28


# logo = io.imread("/home/dell/Documents/TEO/ai-python/digit_prediction-master/images_test/savedSimpleImg-220.jpg", as_grey=True)
# cv2.imshow("logo" ,  logo)
# cv2.waitKey(0)
# logo1 = misc.imresize(logo, (28,28))
# image_predict_image(logo1)
# print(logo.shape)       #28*28


# image_color_to_gray_size
# image_color_to_gray_size("/home/dell/Documents/TEO/ai-python/digit_prediction-master/savedSimpleImg-391.jpg")



