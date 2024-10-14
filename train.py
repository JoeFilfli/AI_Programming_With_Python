import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

def train(model, trainloader, validloader, criterion, optimizer, device, epochs):
    '''Function to train the model.'''
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation after each epoch
        model.eval()
        val_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                val_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {val_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader)*100:.2f}%")
        model.train()

def main():
    parser = argparse.ArgumentParser(description="Train a neural network.")
    
    # Add command-line arguments
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or resnet18).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    args = parser.parse_args()
    
    # Load the dataset
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')

    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    # Load the model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    else:
        print("Model architecture not recognized. Exiting.")
        return
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the classifier
    classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(args.hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    if args.arch == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)
    
    # Set device to GPU or CPU
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Train the model
    train(model, trainloader, validloader, criterion, optimizer, device, args.epochs)
    
    # Save the checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch': args.arch,
                  'hidden_units': args.hidden_units,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': args.epochs}
    
    torch.save(checkpoint, 'checkpoint.pth')
    print("Model saved to checkpoint.pth")

if __name__ == '__main__':
    main()

# python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 4096 --epochs 10 --gpu
