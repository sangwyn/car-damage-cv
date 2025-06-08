import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import segmentation_models_pytorch as smp
from metric import get_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_parts, model_dmg = None, None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_damage_model(model_path = './model_damage.pth'):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",    # !!!!!!!!
        in_channels=3,
        classes=9,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16)
    )

    torch.manual_seed(11)
    model = model.to(device)
    # model_path = os.path.join(WSD, './checkpoints/baseline/epoch_101.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    return model.eval()

def get_parts_model(model_path = './model_parts.pth'):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",    # !!!!!!!!
        in_channels=3,
        classes=13,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16)
    )

    torch.manual_seed(11)
    model = model.to(device)
    # model_path = os.path.join(WSD, './checkpoints/baseline/epoch_101.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    return model.eval()

def get_grade(img_url):
    model_parts = get_parts_model()
    model_dmg = get_damage_model()
    
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mask_parts = model_parts(img_tensor)
            mask_dmg = model_dmg(img_tensor)

            output, _  = get_metric(mask_parts, mask_dmg)

            prediction = torch.argmax(output, dim=1).item() + 1
            
        return min(max(prediction, 1), 5)  # Clamp between 1-5
    except Exception as e:
        print(f"Error processing image: {e}")
        return None