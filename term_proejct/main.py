import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from src.utils import count_acc, Averager, csv_write, square_euclidean_metric
from model import Prototypical_Network, main_network

from src.test_dataset import CUB as Test_Dataset
from src.test_sampler import Test_Sampler

" User input value "
TOTAL = 200000  # total step of training
PRINT_FREQ = 100  # frequency of print loss and accuracy at training step
VAL_FREQ = 1000  # frequency of model eval on validation dataset
SAVE_FREQ = 1000  # frequency of saving model€
TEST_SIZE = 200  # fixed

" fixed value "
VAL_TOTAL = 100

def Test_phase(model, args, k):
    model.eval()

    csv = csv_write(args)

    dataset = Test_Dataset(args.dpath)
    test_sampler = Test_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    test_loader = DataLoader(dataset=dataset, batch_sampler=test_sampler, num_workers=0, pin_memory=True)

    print('Test start!')
    for i in range(TEST_SIZE):
        for episode in test_loader:
            data = episode.cuda()

            data_shot, data_query = data[:k], data[k:]

            """ TEST Method """
            """ Predict the query images belong to which classes
            
            At the training phase, you measured logits. 
            The logits can be distance or similarity between query images and 5 images of each classes.
            From logits, you can pick a one class that have most low distance or high similarity.
            
            ex) # when logits is distance
                pred = torch.argmin(logits, dim=1)
            
                # when logits is prob
                pred = torch.argmax(logits, dim=1)
                
            pred is torch.tensor with size [20] and the each component value is zero to four
            """
            logits = model(data_shot, data_query, args.nway, args.kshot, args.query)
            pred = torch.argmax(logits, dim=1)
            
            # save your prediction as StudentID_Name.csv file
            csv.add(pred)

    csv.close()
    print('Test finished, check the csv file!')
    exit()

def train(args):
    # the number of N way, K shot images
    k = args.nway * args.kshot

    # Train data loading
    dataset = Dataset(args.dpath, state='train')
    train_sampler = Train_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    data_loader = DataLoader(dataset=dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

    # Validation data loading
    val_dataset = Dataset(args.dpath, state='val')
    val_sampler = Sampler(val_dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    val_data_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)

    """ TODO 1.a """
    " Make your own model for Few-shot Classification in 'model.py' file."

    # model setting
    # model = Prototypical_Network()
    model = main_network()

    """ TODO 1.a END """

    # pretrained model load
    if args.restore_ckpt is not None:
        state_dict = torch.load(args.restore_ckpt)
        model.load_state_dict(state_dict)

    model.cuda()
    model.train()

    if args.test_mode == 1:
        Test_phase(model, args, k)

    """ TODO 1.b (optional) """
    " Set an optimizer or scheduler for Few-shot classification (optional) "

    # Default optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay= 0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr= 0.001, weight_decay = 0.1) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000], gamma= 0.1)
    
    # mean squared error
    mse = nn.MSELoss()

    """ TODO 1.b (optional) END """

    tl = Averager()  # save average loss
    ta = Averager()  # save average accuracy

    # training start
    print('train start')
    for i in range(TOTAL):
        for episode in data_loader:
            optimizer.zero_grad()
            
            data, label = [_.cuda() for _ in episode]  # load an episode

            # split an episode images and labels into shots and query set
            # note! data_shot shape is ( nway * kshot, 3, h, w ) not ( kshot * nway, 3, h, w )
            # Take care when reshape the data shot
            data_shot, data_query = data[:k], data[k:]

            label_shot, label_query = label[:k], label[k:]
            label_shot = sorted(list(set(label_shot.tolist()))) # list type, size: nway

            # convert labels into 0-4 values
            label_query = label_query.tolist()
            labels = []
            for j in range(len(label_query)):
                label = label_shot.index(label_query[j])
                labels.append(label)
            labels = torch.tensor(labels).cuda()

            """ TODO 2 ( Same as above TODO 2 ) """
            """ Train the model 
            Input:
                data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                            be careful when using torch.reshape or .view functions
                data_query : torch.tensor, query images, [args.query, 3, h, w]
                labels : torch.tensor, labels of query images, [args.query]
            output:
                loss : torch scalar tensor which used for updating your model
                logits : A value to measure accuracy and loss
            """
            
            '''
            # Prototypical network
            logits = model(data_shot, data_query, args.nway, args.kshot, args.query)
            
            logits_softmax = F.log_softmax(-logits, dim=1)
        
            labels_unsqez = torch.unsqueeze(labels, 1)
        
            loss = -torch.gather(logits_softmax, 1, labels_unsqez).mean()
            '''
            
            
            # relation network
            logits = model(data_shot, data_query, args.nway, args.kshot, args.query)
            
            one_hot_labels = torch.zeros(args.query, args.nway).scatter_(1, labels.cpu().view(-1, 1), 1)
            loss = mse(logits, one_hot_labels.cuda())
            

            """ TODO 2 END """

            acc = count_acc(logits, labels)

            tl.add(loss.item())
            ta.add(acc)

            loss.backward()
            optimizer.step()
            scheduler.step()

            proto = None; logits = None; loss = None

        if (i+1) % PRINT_FREQ == 0:
            print('train {}, loss={:.4f} acc={:.4f}'.format(i+1, tl.item(), ta.item()))

            # initialize loss and accuracy mean
            tl = None
            ta = None
            tl = Averager()
            ta = Averager()

        # validation start
        if (i+1) % VAL_FREQ == 0:
            print('validation start')
            model.eval()
            with torch.no_grad():
                vl = Averager()  # save average loss
                va = Averager()  # save average accuracy
                for j in range(VAL_TOTAL):
                    for episode in val_data_loader:
                        data, label = [_.cuda() for _ in episode]

                        data_shot, data_query = data[:k], data[k:] # load an episode

                        label_shot, label_query = label[:k], label[k:]
                        label_shot = sorted(list(set(label_shot.tolist())))

                        label_query = label_query.tolist()

                        labels = []
                        for j in range(len(label_query)):
                            label = label_shot.index(label_query[j])
                            labels.append(label)
                        labels = torch.tensor(labels).cuda()

                        """ TODO 2 ( Same as above TODO 2 ) """
                        """ Train the model 
                        Input:
                            data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                                        be careful when using torch.reshape or .view functions
                            data_query : torch.tensor, query images, [args.query, 3, h, w]
                            labels : torch.tensor, labels of query images, [args.query]
                        output:
                            loss : torch scalar tensor which used for updating your model
                            logits : A value to measure accuracy and loss
                        """

                        '''
                        # Prototypical network
                        logits = model(data_shot, data_query, args.nway, args.kshot, args.query)
                        
                        logits_softmax = F.log_softmax(-logits, dim=1)
                    
                        labels_unsqez = torch.unsqueeze(labels, 1)
                    
                        loss = -torch.gather(logits_softmax, 1, labels_unsqez).mean()
                        '''
                        
                        # relation network
                        logits = model(data_shot, data_query, args.nway, args.kshot, args.query)
                        
                        one_hot_labels = torch.zeros(args.query, args.nway).scatter_(1, labels.cpu().view(-1, 1), 1)
                        loss = mse(logits, one_hot_labels.cuda())

                        """ TODO 2 END """

                        acc = count_acc(logits, labels)

                        vl.add(loss.item())
                        va.add(acc)

                        proto = None; logits = None; loss = None

                print('val accuracy mean : %.4f' % va.item())
                print('val loss mean : %.4f' % vl.item())

                # initialize loss and accuracy mean
                vl = None
                va = None
                vl = Averager()
                va = Averager()
            model.train()

        if (i+1) % SAVE_FREQ == 0:
            PATH = 'checkpoints/%d_%s.pth' % (i + 1, args.name)
            torch.save(model.state_dict(), PATH)
            print('model saved, iteration : %d' % i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model', help="name your experiment")
    parser.add_argument('--dpath', '--d', default='./dataset/CUB_200_2011/CUB_200_2011', type=str,
                        help='the path where dataset is located')
    parser.add_argument('--restore_ckpt', default='./checkpoints/30000_model.pth', type=str, help="restore checkpoint")
    # parser.add_argument('--restore_ckpt', type=str, help="restore checkpoint")
    parser.add_argument('--nway', '--n', default=5, type=int, help='number of class in the support set (5 or 20)')
    parser.add_argument('--kshot', '--k', default=5, type=int,
                        help='number of data in each class in the support set (1 or 5)')
    parser.add_argument('--query', '--q', default=20, type=int, help='number of query data')
    parser.add_argument('--ntest', default=100, type=int, help='number of tests')
    parser.add_argument('--gpus', type=int, nargs='+', default=0)
    parser.add_argument('--test_mode', type=int, default=1, help="if you want to test the model, change the value to 1")

    args = parser.parse_args()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    torch.cuda.set_device(args.gpus)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print ('Available devices ', torch.cuda.device_count())
    # print ('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(device))
    
    train(args)

