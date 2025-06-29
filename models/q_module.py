import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def QNODE(inputs, weights_0, weights_1):

    measure_set = [0, 2]
    groups = [[0, 1, 2, 3]]
    # group the channels
    for l in range(1):
        for ws in groups:

            qml.RY(inputs[:, ws[0]], wires = ws[0])
            qml.RY(inputs[:, ws[1]], wires = ws[1])
            qml.RY(inputs[:, ws[2]], wires = ws[2])
            qml.RY(inputs[:, ws[3]], wires = ws[3])

            qml.RY(weights_0[0, l, ws[0]], wires = ws[0])
            qml.RY(weights_0[0, l, ws[1]], wires = ws[1])
            qml.RY(weights_0[0, l, ws[2]], wires = ws[2])
            qml.RY(weights_0[0, l, ws[3]], wires = ws[3])

            qml.IsingXX(weights_1[l, ws[0]], wires = [ws[0], ws[1]])
            qml.IsingXX(weights_1[l, ws[1]], wires = [ws[2], ws[3]])

            qml.RX(weights_0[1, l, ws[0]], wires = ws[0])
            qml.RX(weights_0[1, l, ws[1]], wires = ws[1])
            qml.RX(weights_0[1, l, ws[2]], wires = ws[2])
            qml.RX(weights_0[1, l, ws[3]], wires = ws[3])

            qml.IsingXX(weights_1[l, ws[2]], wires = [ws[1], ws[2]])
            qml.IsingXX(weights_1[l, ws[3]], wires = [ws[0], ws[3]])
            
            qml.RY(weights_0[2, l, ws[0]], wires = ws[0])
            qml.RY(weights_0[2, l, ws[1]], wires = ws[1])
            qml.RY(weights_0[2, l, ws[2]], wires = ws[2])
            qml.RY(weights_0[2, l, ws[3]], wires = ws[3])

            qml.MultiControlledX(control_wires=[ws[0],ws[1]], wires=ws[2], control_values="10")
            qml.MultiControlledX(control_wires=[ws[1],ws[2]], wires=ws[3], control_values="10")
            qml.MultiControlledX(control_wires=[ws[2],ws[3]], wires=ws[0], control_values="10")
            qml.MultiControlledX(control_wires=[ws[3],ws[0]], wires=ws[1], control_values="10")

    exp_vals_z = [qml.expval(qml.PauliZ(w)) for w in measure_set]

    return exp_vals_z

class Qmodule(nn.Module):
    def __init__(self, batch_size = 4, out_height = 128, out_width = 128, mode = "bilinear"):
        super().__init__()
        self.weight_shapes = {"weights_0": (3, 1, 4), "weights_1": (1, 4)}
        self.qlayer1 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer5 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer6 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer7 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.qlayer8 = qml.qnn.TorchLayer(QNODE, self.weight_shapes)
        self.max = nn.AdaptiveAvgPool2d((2, 2))
        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.out_height = out_height
        self.out_width = out_width
        self.batch_size = batch_size
        self.mode = mode
        

    def forward(self, x):
        device = x.device
        B = x.shape[0]
        x = 0.4 * self.avg(x) + 0.6 * self.max(x)
        x = x.view(B, -1, 4)
        # [b, c, 4] -> [b, c, 4]
        x = torch.cat((self.qlayer1(x), self.qlayer2(x), self.qlayer3(x), self.qlayer4(x),\
                       self.qlayer5(x), self.qlayer6(x), self.qlayer7(x), self.qlayer8(x)), 2)
        x = x.view(B, -1, 4, 4)
        x = F.interpolate(
            x.to(device), 
            size=(self.out_height, self.out_width), 
            mode=self.mode, 
            align_corners=False
        ) if self.mode in ('bilinear', 'bicubic') else self.upsample_conv(x.to(device))
        return x		   # b, c, 128, 128, fixed 
    
if __name__ == "__main__":
    model = Qmodule()
    inputs = torch.randn(4, 88, 9527, 64)
    outputs = model(inputs)
    print(outputs.shape)  
