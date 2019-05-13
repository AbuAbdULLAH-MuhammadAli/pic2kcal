from nn.models.res_net import ResNet

if __name__ == '__main__':

    model = ResNet()
    net = model.get_model_on_device()

    print(net)


