import torch
from torchvision import models
from PIL import Image
import numpy as np
import argparse
import json

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model. '''
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True) if checkpoint['arch'] == 'vgg16' else models.resnet18(pretrained=True)
    
    if checkpoint['arch'] == 'vgg16':
        model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(checkpoint['hidden_units'], 102),
                                         nn.LogSoftmax(dim=1))
    else:
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, checkpoint['hidden_units']),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(checkpoint['hidden_units'], 102),
                                 nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class of an image using a trained model. '''
    model.eval()
    
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
        ps = torch.exp(output)
    
    probs, indices = ps.topk(topk)
    probs = probs.numpy().flatten()
    indices = indices.numpy().flatten()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image.")
    
    parser.add_argument('image_path', type=str, help='Path to image for prediction.')
    parser.add_argument('checkpoint', type=str, help='Path to saved model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions.')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction.')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Set device to GPU or CPU
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Make the prediction
    probs, classes = predict(args.image_path, model, topk=args.top_k)
    
    # Load category names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(c)] for c in classes]
    
    print("Top K classes and probabilities:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob}")
    
if __name__ == '__main__':
    main()


#python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
