import torchvision.transforms as transforms


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to (224, 224)
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize images
        ]
    )

    return transform(image)
