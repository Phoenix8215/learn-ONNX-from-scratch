import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
    def forward(self, x):
        x = self.convs1(x)
        x1 = self.convs2(x)
        x2 = self.convs3(x)
        x = x1 + x2
        x = self.convs4(x)
        return x

model = Model()
input = torch.randn(1, 3, 20, 20)

torch.onnx.export(model, input, 'whole_model.onnx')