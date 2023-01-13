# import
import argparse
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
def rep(training_weights='cfg/training/yolov7_training.pt',
        deploy_yaml='cfg/deploy/yolov7.yaml',
        deploy_weights='cfg/deploy/yolov7.pt'):
    
    device = select_device('0', batch_size=1)
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(training_weights, map_location=device)
    # reparameterized model in cfg/deploy/*.yaml
    model = Model(deploy_yaml, ch=3, nc=10+13).to(device)

    with open(deploy_yaml) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    
    print(f'model.names:{model.names}, model.nc:{model.nc}')

    # reparametrized YOLOR
    for i in range((model.nc+5)*anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, deploy_weights)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_weights', type=str, default='/project/train/models/exp/weights/last.pt', help='weights path')
    parser.add_argument('--deploy_yaml', type=str, default='/project/train/cfg/deploy/vehicle_yolov7_deploy.yaml', help='image size')  # height, width
    parser.add_argument('--deploy_weights', type=str, default='/project/train/models/exp/weights/deploy_yolov7_last.pt', help='batch size')
    opt = parser.parse_args()
    
    # training_weights='/project/train/models/exp2/weights/best.pt'
    # deploy_yaml='cfg/deploy/vehicle_yolov7_deploy.yaml'
    # deploy_weights='/project/train/models/exp2/weights/deploy_yolov7.pt'
    
    rep(training_weights=opt.training_weights,
        deploy_yaml=opt.deploy_yaml,
        deploy_weights=opt.deploy_weights)