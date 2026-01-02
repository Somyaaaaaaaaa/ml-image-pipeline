from data_loader import load_data
from preprocessing import preprocess
from model import SimpleCNN
from train import train
from evaluate import evaluate


def main():
    images, labels = load_data() #load data
    images = preprocess(images) #preprocess data
    model = SimpleCNN(num_classes=2) #build model
    model = train(model, images, labels, epochs = 5) #train model
    metrics = evaluate(model, images, labels) #evaluate model

    print ("Final Metrics:", metrics)

if __name__ == "__main__":
    main()
