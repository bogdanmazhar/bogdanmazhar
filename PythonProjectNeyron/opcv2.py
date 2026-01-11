import cv2

img = cv2.imread("road2.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

car_data = cv2.CascadeClassifier('haarcascade_cars.xml')
stop_data = cv2.CascadeClassifier('haarcascade_stopsign.xml')

stop_coords = []
car_coords = []

try:
    car_coords = car_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
    print(car_coords)
except:
    print('Машины не найдены')

img_height, img_width, img_channels = img.shape
left_border = img_width / 2
right_border = img_width

def check_forward(car_coords, stop_coords):
    if len(stop_coords) != 0:
        return False
    elif len(car_coords) == 0:
        return True
    else:
        for (x, y, width, height) in car_coords:
            if x > left_border and x + width < right_border:
                if width / img_width > 0.15:
                    return False
        return True

print(check_forward(car_coords, stop_coords))


try:
    stop_coords = stop_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
except:
    print('Знаки СТОП не найдены')

def check_forward(stop_coords):
    if len(stop_coords) != 0:
        return False
    else:
        return True

print('Движение разрешено:', check_forward(stop_coords))