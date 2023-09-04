from Environment import *
from Network.model.hrnet3D import HRNet
from Network.model.swin_transform import swin_tiny_patch4_window8_256, swin_base_patch4_window8_256
from Network.model.FCN_resnet3d import fcn_resnet34
from Network.model.unet3d import UNet3D
from Network.model.ASPP_unet3d import UNet3D
from Network.model.FCN import FCN
from Network.dataset_VISoR import VISoR_dataset, RandomGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from Network.utils import ConfusionMatrix, DiceLoss, weights_init, init_weights, get_params_groups, create_lr_scheduler
from torch.autograd import Variable
from Parameter import *

warnings.filterwarnings("ignore")
def train(net, checkpoint, pre_train, device):
    if pre_train:
        weight = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict(weight, strict=False)
        print('net pre-train !')
    net.train()
    logging.basicConfig(filename=Network['save_checkpoint'] + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    img_size = Preprocessing['cut_size']
    base_lr = Network['lr']
    max_epochs = Network['max_epoch']
    num_classes = Network['num_classes']
    data_path = Network['train_data']
    list_dir = os.path.join(Network['save_data_list'], 'lists')
    save_checkpoint = Network['save_checkpoint']
    train_data = VISoR_dataset(base_dir=data_path, list_dir=list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[img_size, img_size])]))
    val_data = VISoR_dataset(base_dir=data_path, list_dir=list_dir, split="valid",
                                transform=transforms.Compose([RandomGenerator(output_size=[img_size, img_size])]))

    train_loader = DataLoader(train_data, batch_size=Network['train_batch_size'], shuffle=Network['train_shuffle'], 
                                num_workers=Network['train_num_workers'], pin_memory=Network['pin_memory'])
    valid_loader = DataLoader(val_data, batch_size=Network['valid_batch_size'], shuffle=Network['valid_shuffle'], num_workers=Network['valid_num_workers'])

    print("The length of train set is: {}".format(len(train_loader)))
    print("The length of valid set is: {}".format(len(valid_loader)))
    # params_to_optimize = [
    #     {"params": [p for p in net.parameters() if p.requires_grad]},
    # ]
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=Optimizer['momentum'], weight_decay=Optimizer['SGD_weight_decay'])
    # optimizer = optim.Adam(net.parameters(), lr= base_lr, weight_decay=Optimizer['Adam_weight_decay'])
    
    # pg = get_params_groups(net, weight_decay=5e-2)
    # optimizer = optim.AdamW(pg, lr=base_lr, weight_decay=Optimizer['AdmaW_weight_decay'])
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), max_epochs, warmup=True)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    iter_num = 0
    validate_every_n_epoch = Network['valid_save_every_n_epoch']
    loss_ratio = Network['loss_ratio']
    max_epoch = max_epochs
    max_iterations = max_epochs * len(train_loader)   
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            image_batch, label_batch = Variable(image_batch), Variable(label_batch)
            pred = net(image_batch)
            # pred = pred['out']
            loss_ce = criterion(pred, label_batch[:].long())
            loss_dice = dice_loss(pred, label_batch, softmax=False)
            loss = loss_ratio * loss_ce + (1 - loss_ratio) * loss_dice
            # loss = criterion(pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            lr_ = optimizer.param_groups[0]["lr"]
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # logging.info('iteration %d : loss : %f, lr: %f' % (iter_num, loss.item(), lr_))
            logging.info('iteration %d : loss : %f, loss_ce : %f' % (iter_num, loss.item(), loss_ce.item()))

        if epoch_num % validate_every_n_epoch ==0:
            confmat = ConfusionMatrix(num_classes)
            val_loader = tqdm(valid_loader, desc="Validate")
            val_iter_num = 0
            net.eval()
            for i, val_data in enumerate(val_loader):
                val_image, val_label = val_data['image'], val_data['label']
                val_image, val_label = val_image.to(device), val_label.to(device)
                with torch.no_grad():
                    val_out = net(val_image)
                    # val_out = val_out['out']
                    val_out = torch.softmax(val_out, dim=1)
                    confmat.update(val_label.squeeze().flatten(), torch.argmax(val_out, dim=1).flatten())
                val_iter_num = val_iter_num + 1
                logging.info(confmat)
                # logging.info('val_iteration %d : val_loss : %f, val_loss_ce: %f' % (val_iter_num, val_loss.item(), val_loss_ce.item()))

        save_interval = 5  # int(max_epoch/5)

        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(net.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(save_checkpoint, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(net.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"

if __name__ == '__main__':
    pre_train = Network['pre_train']
    checkpoint_path = os.path.join(pre_train, 'epoch_9.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # net = FCN(in_channels=1, num_classes=Network['num_classes']).to(device)
    net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=Network['num_classes']).to(device)
    # net = fcn_resnet34(num_classes=Network['num_classes']).to(device)
    # net = UNet3D(in_channel=1, n_classes=Network['num_classes']).to(device)
    # net = HRNet(num_classes=Network['num_classes']).to(device)
    net = net.apply(weights_init)
    # net.init_weights()
    print('net initialization succeeds !')
    train(net, checkpoint_path, pre_train, device)