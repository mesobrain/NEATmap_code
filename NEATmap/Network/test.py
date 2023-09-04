from Environment import *
from Parameter import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Network.dataset_VISoR import VISoR_dataset
from Network.utils import test_single_volume
from Network.model.hrnet3D import HRNet
from Network.model.ASPP_unet3d import UNet3D
from Network.model.FCN import FCN
from Network.model.FCN_resnet3d import fcn_resnet34
from Network.model.unet3d import UNet3D
from Network.model.swin_transform import swin_tiny_patch4_window8_256, swin_base_patch4_window8_256

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='', help='root dir for validation volume data') 
parser.add_argument('--dataset', type=str,
                    default='VISoR', help='experiment_name')
parser.add_argument('--max_epochs', type=int,
                    default=Network['max_epochs'], help='maximum epoch number to train')
parser.add_argument('--num_classes', type=int,
                    default=Network['num_classes'], help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--model_name', type=str,
                    default='swin', help='model_name')

parser.add_argument('--batch_size', type=int, default=Network['test_batch_size'], help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=Preprocessing['cut_size'], help='input patch size of network input')
parser.add_argument('--is_savetif', action="store_true", default=True, help='whether to save results during inference')
parser.add_argument('--test_id', type=int, default=30, help='test the 30th brain slice')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as tif!')
parser.add_argument('--base_lr', type=float,  default=Network['lr'], help='segmentation network learning rate')
parser.add_argument('--patch_csv_root', type=str,  default=Network['save_patch_metrics'], help='The path to save the patch evaluation result csv')
parser.add_argument('--slice_csv_root', type=str,  default=Network['save_metrics'], help='The path to save the slice evaluation result csv')
args = parser.parse_args(args=[])

def test(args, model, test_save_path=None):
    test_data = args.Dataset(base_dir=args.volume_path, split="Z{:05d}_test".format(args.test_id), list_dir=args.list_dir)

    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = []
    invalid_num = 0
    patch_csv_name = args.model_name + '_patch_results.csv'
    patch_csv_root = args.patch_csv_root
    os.makedirs(patch_csv_root, exist_ok=True)
    patch_csv_save_path = os.path.join(patch_csv_root, patch_csv_name)
    patch_list_name = ['Dice', 'Hd95', 'Jc', 'Sst']
    with open(patch_csv_save_path, 'w+', newline='')as f:
        patch_csv_write = csv.writer(f, dialect='excel')
        patch_csv_write.writerow(patch_list_name)
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            # h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i, confmat = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                        test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            # metric_list += np.array(metric_i)
            patch_writelines = [metric_i[0][0], metric_i[0][1], metric_i[0][2], metric_i[0][3]]
            patch_csv_write.writerow(patch_writelines)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_jc %f mean_sst %f' % (i_batch, case_name, 
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
                        np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]
                        ))
            if np.mean(metric_i, axis=0)[0] == 0:
                pass
            else:
                metric_list.append(metric_i)
    confmat.reduce_from_all_processes()
    logging.info(confmat)
    # metric_mean_list = metric_list / (len(testloader) - invalid_num)
    metric_mean_list = np.mean(metric_list, axis=0)
    metric_std_list = np.std(metric_list, axis=0)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_jc %f mean_sst %f' % (i, metric_mean_list[i-1][0], metric_mean_list[i-1][1], 
                                                                                            metric_mean_list[i-1][2], metric_mean_list[i-1][3]))
                                                                                      
    performance = np.mean(metric_mean_list, axis=0)[0]
    mean_hd95 = np.mean(metric_mean_list, axis=0)[1]
    mean_jc = np.mean(metric_mean_list, axis=0)[2]
    mean_sst = np.mean(metric_mean_list, axis=0)[3]
    Mean_list = [performance, mean_hd95, mean_jc, mean_sst]

    std_dice = metric_std_list[0][0]
    std_hd95 = metric_std_list[0][1]
    std_jc = metric_std_list[0][2]
    std_sst = metric_std_list[0][3]
    Std_list = [std_dice, std_hd95, std_jc, std_sst]

    ste_dice = std_dice / math.sqrt((len(testloader) - invalid_num))
    ste_hd95 = std_hd95 / math.sqrt((len(testloader) - invalid_num))
    ste_jc = std_jc / math.sqrt((len(testloader) - invalid_num))
    ste_sst = std_sst / math.sqrt((len(testloader) - invalid_num))
    Ste_list = [ste_dice, ste_hd95, ste_jc, ste_sst]

    csv_name = args.model_name + '_results.csv'
    csv_root = args.slice_csv_root
    os.makedirs(csv_root, exist_ok=True)
    csv_save_path = os.path.join(csv_root, csv_name)
    list_name = ['Metric_name','Mean', 'Std', 'Ste']
    metric_name = ['Dice', 'Hd95', 'Jc', 'Sst']
    with open(csv_save_path, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(list_name)
        for num in range(len(metric_name)):
            writelines = [metric_name[num], Mean_list[num], Std_list[num], Ste_list[num]]
            csv_write.writerow(writelines)

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_jc: %f mean_sst: %f' % (performance, mean_hd95, mean_jc, mean_sst))
    logging.info('Testing performance in best val model: std_dice : %f std_hd95 : %f std_jc: %f std_sst: %f' % (std_dice, std_hd95, std_jc, std_sst))
    logging.info('Testing performance in best val model: ste_dice : %f ste_hd95 : %f ste_jc: %f ste_sst: %f' % (ste_dice,ste_hd95, ste_jc, ste_sst))

    return "Testing Finished!"

if __name__ == "__main__":

    dataset_config = {
        'VISoR': {
            'Dataset': VISoR_dataset,
            'volume_path': Test_config['test_save_path'],
            'list_dir': '/home/weijie/3dHRNet/lists',
            'num_classes': Network['num_classes'],
            'z_spacing': Network['z_spacing'],
        },
    }
    dataset_name = args.dataset
    args.exp = dataset_name + str(args.img_size)
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.test_id = Test_config['test_id']
    args.is_savetif = True

    # name the same snapshot defined in train script!
    snapshot_path = Network['snapshot_path']

    # net = HRNet(num_classes=args.num_classes, width=Model['hrnet_width']).cuda()
    # net = UNet3D(in_channel=Network['in_channels'], n_classes=args.num_classes).cuda()
    # net = FCN(in_channels=Network['in_channels'], num_classes=args.num_classes).cuda()
    # net = fcn_resnet34(num_classes=args.num_classes).cuda()
    net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=args.num_classes).cuda()
    # net = swin_base_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=args.num_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(Network['save_prediction_path'], 'brain_predications_' + args.test_id + '_' + args.model_name + '_epoch' + str(args.max_epochs))
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    test(args, net, test_save_path)