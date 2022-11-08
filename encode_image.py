import base64

with open('/home/marin/Desktop/Dataset/PlantVillageResized/testing/Tomato Yellow Leaf Curl Virus/10293c1b-da1e-4b3a-821e-4f71c54c2733___YLCV_NREC 2751.JPG', "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    print(encoded_string)