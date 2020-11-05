import torch
from blazeface import BlazeFace
class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript,self).__init__()
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = BlazeFace().to(gpu)
        net.load_weights("blazeface.pth")
        self.net = net
    def forward(self, x):
        return self.net.forward(x)
x = torch.rand(1,3,224,224).float().cuda()
pt_model = PyTorch_to_TorchScript().eval()
traced_script_module = torch.jit.trace(pt_model, x)
traced_script_module.save("model.pt")