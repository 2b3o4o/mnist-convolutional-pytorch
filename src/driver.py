from train import Trainer
from model import ConvNet

def main():
    model = ConvNet(hidden_size=64, output_size=10)
    trainer = Trainer(model)
    trainer.train(30)
    trainer.evaluate_accuracy()

if __name__ == "__main__":
    main()
