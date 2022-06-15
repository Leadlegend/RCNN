from .svm import SVMModel
from .cnn import CNNModel, FtCNNModel
from .reg import RegModel
from .criterions import CNNLoss, RegLoss, HingeLoss

model_factory = {
    'svm': SVMModel,
    'alex': CNNModel,
    'finetune': FtCNNModel,
    'reg': RegModel,
}

criterion_factory = {
    'alex': CNNLoss,
    'finetune': CNNLoss,
    'svm': HingeLoss,
    'reg': RegLoss,
}
