from PIL import Image
init_image = Image.open('1.png').convert('RGB')
init_image.save('RGB.png')
