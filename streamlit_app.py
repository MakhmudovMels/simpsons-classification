import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
from torchvision import transforms

# Определение размеров изображения
INFER_WIDTH = 224
INFER_HEIGHT = 224

# Статистика нормализации для ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Преобразование изображения в PyTorch тензор и нормализация
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

# Определение устройства для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка JIT модели
best_model = torch.jit.load('models/best_model.pt', map_location=DEVICE)

# Загрузка label encoder
label_encoder = pickle.load(open("label-encoder/label_encoder.pkl", 'rb'))


def infer_image(image):
    """Предсказывание класса изображения с помощью загруженной модели."""
    
    # Подгоняем изображение под вход модели
    image = image.resize((INFER_HEIGHT, INFER_WIDTH))

    # Преобразование изображения в PyTorch тензор и нормализация
    inputs = transform(image)

    # Прогон изображения через модель
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        best_model.eval()
        logit = best_model(inputs.unsqueeze(0))
        probs_im = torch.nn.functional.softmax(logit, dim=-1).numpy()

    # Получаем предсказанный класс
    y_pred = np.argmax(probs_im)
    pred_class = label_encoder.classes_[y_pred]

    return pred_class

def display_image(image):
    """Отображение изображения."""
    st.image(image, width=70)

def upload_image(label):
    """Загрузка изображения."""
    uploaded_file = st.file_uploader(label, type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        return Image.open(uploaded_file)
    return None

def main():
    st.set_page_config(
        page_title="Classification of the Simpsons",
        )

    st.title('Classification of the Simpsons')

    st.write('As you know, The Simpsons series has been on \
             TV for more than 25 years, and during this time \
             a lot of video material has accumulated. The \
             characters have changed along with changing \
             graphics technologies, and Homer Simpson 2018 \
             is not very similar to Homer Simpson 1989. This \
             application is designed to classify the characters \
             living in Springfield.')

    # Загрузка изображения
    image = upload_image('Upload an image')

    # Проверка, что изображение загружено
    if image is not None:
        # Отображение исходного изображения
        display_image(image)

        # Делаем predict
        pred_class = infer_image(image)

        # Отображаем предсказанный класс
        st.success(f'Predicted class : {pred_class}')


if __name__ == '__main__':
    main()
