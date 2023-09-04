from Environment import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Network.dataset_whole_brain import Whole_brain_dataset
from Network.utils import test_each_brain
from Network.model.swin_transform import swin_tiny_patch4_window8_256, swin_base_patch4_window8_256
from Parameter import *

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
parser.add_argument('--channel', type=str, default=Channels['staining'], help='selecting the channel for reasoning about the whole brain')
parser.add_argument('--is_savetif', action="store_true", default=True, help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as tif!')
parser.add_argument('--base_lr', type=float,  default=Network['lr'], help='segmentation network learning rate')
parser.add_argument('--patch_csv_root', type=str,  default=Network['save_patch_metrics'], help='The path to save the patch evaluation result csv')
parser.add_argument('--slice_csv_root', type=str,  default=Network['save_metrics'], help='The path to save the slice evaluation result csv')

args = parser.parse_args(args=[])

def infer_whole_brain(args, model, ind, test_save_path=None):
    test_data = args.Dataset(base_dir=args.volume_path, split=args.name_list, list_dir=args.list_dir)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, case_name = sampled_batch["image"], sampled_batch['case_name'][0]
        test_each_brain(image, model, patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

    return logging.info("Testing Finished {}".format(ind))

if __name__ == "__main__":
    data_path = Network['whole_brain_path']
    test_path = os.path.join(data_path, 'PatchImage_' + args.channel)
    for ind in range(1, len(os.listdir(test_path)) + 1):
        dataset_config = {
            'VISoR': {
                'Dataset': Whole_brain_dataset,
                'volume_path': test_path + '/patchimage{}'.format(ind),
                'list_dir': Network['whole_brain_list'],
                'name_list': 'Z{:05d}_test'.format(ind),
                'z_spacing': Network['z_spacing'],
            },
        }
        dataset_name = args.dataset
        args.exp = dataset_name + str(args.img_size)
        args.volume_path = dataset_config[dataset_name]['volume_path']
        args.Dataset = dataset_config[dataset_name]['Dataset']
        args.list_dir = dataset_config[dataset_name]['list_dir']
        args.name_list = dataset_config[dataset_name]['name_list']
        args.z_spacing = dataset_config[dataset_name]['z_spacing']
        args.is_pretrain = True

        # name the same snapshot defined in train script!
        snapshot_path = Network['snapshot_path']

        net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=args.num_classes).cuda()

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
            args.test_save_dir = data_path + '/whole_predications_' + args.channel + 'brain_predications_{}_swin_epoch10'.format(ind)
            test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None
        infer_whole_brain(args, net, ind, test_save_path)