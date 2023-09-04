from Environment import *
from Parameter import *
def data_list(train_data, ratio, test_id, save_list, test_single=None, test_whole_brain=None):

    path = train_data
    name_list = os.listdir(path)
    name = []
    for vlaue in name_list:
        protions = vlaue.split('.')
        line = '{}\n'.format(protions[0])
        name.append(line)
    total_num = len(name)
    train_num = int(len(name)*ratio)
    train_index = name[0:train_num]
    valid_index = name[train_num:total_num]
    with open(os.path.join(save_list, 'lists/train.txt'), 'w') as f:
        f.writelines(train_index)
    with open(os.path.join(save_list, 'lists/valid.txt'), 'w') as v:
        v.writelines(valid_index)

    if test_single is not None:
        path = test_single
        name_list = os.listdir(path)
        name = []
        for vlaue in name_list:
            protions = vlaue.split('.')
            line = '{}\n'.format(protions[0])
            name.append(line)
        with open(os.path.join('lists/Z{:05d}_test.txt'.format(test_id)), 'w') as f:
            f.writelines(name)

    if test_whole_brain is not None:
        whole_brain_path = os.path.join(test_whole_brain, 'whole_brain_test')
        for ind in range(1, len(os.listdir(whole_brain_path)) + 1):
            path = os.path.join('brain_test_{}'.format(ind))
            name_list = os.listdir(path)
            name = []
            for vlaue in name_list:
                protions = vlaue.split('.')
                line = '{}\n'.format(protions[0])
                name.append(line)
            with open(os.path.join(whole_brain_path, 'whole_brain_lists/Z{:05d}_test.txt'.format(ind)), 'w') as f:
                f.writelines(name)

if __name__ == "__main__":
    data_list(train_data=Network['train_data'], ratio=Network['train_valid_ratio'], test_id=Test_config['test_id'],
            save_list=Network['save_data_list'], test_singel=Network['test_slice'], test_whole_brain=Network['test_whole_brain'])