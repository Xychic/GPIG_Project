import classifier
import torch
import os
import sys

def optimFactory(optimizer_class,*args,**kwargs):
    def _optim(parameters):
        return optimizer_class(parameters,*args,**kwargs)
    return _optim

defaultAdamOptim = optimFactory(torch.optim.Adam,lr=0.001, betas=(0.9, 0.999), weight_decay=0)
resnet_base_SGD = optimFactory(torch.optim.SGD, lr=0.1, momentum=0.9, dampening=0, weight_decay=0.0001)
lr_plataeau = optimFactory(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
resnet_SGD = optimFactory(lambda x,y,z: z(y(x)), resnet_base_SGD,lr_plataeau)
#variables dictating training
traintypes = [#augmentation,optimizer,weighted,epoch,limit,accuracy
    [torch.nn.Identity(),resnet_base_SGD,False,10,lambda loss,accu: False,False],
    [non_dist_augments,resnet_SGD,False,1000,classifier.ProgressMade(50,classifier.CoolRate(15,15))]
]
#variables dictating the model
modeltypes = [#class,image_corrector,image_size,other class variables... (e.g. stages,block,starting_channels,reduction)
    [classifier.Resnetish,torch.nn.Identity(),480,[3,4,6,3],classifier.Resnet_block,64,4]
]


def CalcAccu(model,test_data,image_corrector = torch.Identity()):
    model.eval()
    correct = 0
    total = 0
    for x,y in test_data:
        result = model(image_corrector(x))
        types = torch.argmax(result,dim=1)
        correct += torch.sum(types == y)
        total += len(y)
    model.train()
    return correct/total

def interTrain(model,optim,data,image_corrector,augment,losser,epochs,limit,testing_dataset):
    model.train()
    all_loss = []
    all_accu = []
    accu = 0.
    stop = False
    for epoch in range(epochs):
        try:
            for x,y in data:
                x_aug = augment(image_corrector(x))
                #forwards
                result = model(x_aug)
                loss = losser(result,y)
                #backwards
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                all_loss.append(loss.item())
                print("Epoch {}, Loss: {:.4f}".format(epoch + 1, loss.item()))
                if limit(loss.item(),accu):
                    print("limit hit")
                    stop = True
                    break
            if testing_dataset: 
                accu = CalcAccu(model,testing_dataset,image_corrector)
            #display + store for graph
                print("Epoch {}, Loss: {:.4f}, Accu: {:.4f}".format(epoch + 1, loss.item(),accu[0]))
                all_accu.append(accu)
            if stop or limit(loss.item(),accu):
                break
        except KeyboardInterrupt:
            print("interTrain stopped early")
            break
    return (all_loss, all_accu) if testing_dataset else all_loss

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 7:
        print("dataset_filter.py inputdir speciesListFile traintype [modeltype [modelfilesave [modelfileload]]]]")
        sys.exit()
    inp = os.path.abspath(sys.argv[1])
    speciesListFile = os.path.abspath(sys.argv[2])

    if not os.path.isdir(inp):
        raise OSError("Invalid input directory: " + inp)
    elif not os.access(inp,os.R_OK):
        raise OSError("No read permission for input directory: " + inp)
    if not os.path.isfile(speciesListFile):
        raise OSError("Invalid output directory: " + speciesListFile)
    elif not os.access(speciesListFile,os.W_OK):
        raise OSError("No write permission for output directory: " + speciesListFile)
    with open(speciesListFile) as f:
        species_list = f.read().splitlines()

    try:
        traintypeindex = int(sys.argv[3])
        traintype = traintypes[traintypeindex]
    except:
        raise ValueError("Invalid train type: " + sys.argv[3])
    if len(sys.argv) >= 5:
        try:
            modeltypeindex = int(sys.argv[5])
            modeltype = modeltypes[modeltypeindex]
        except:
            raise ValueError("Invalid model type: " + sys.argv[4])
    else:
        modeltype = modeltypes[0]

    model = modeltype[0](len(species_list),*(modeltype[2:]))
    image_corrector = modeltype[1](modeltype[2])
    augments = traintype[1]
    optim = traintype[2](model.parameters())
    weighted = traintype[3]
    epoch = traintype[4]
    limit = traintype[5]
    accuracy = traintype[6]
    
    if len(sys.argv) >= 5:
        modelfile_save = os.path.abspath(sys.argv[4])
        save_dir = os.path.dirname(modelfile_save)
        if not os.access(save_dir, os.W_OK):
            raise OSError("Invalid model file save location: " + modelfile_save)
        if len(sys.argv) >= 6:
            modelfile_load = os.path.abspath(sys.argv[5])
            if not os.path.isfile(modelfile_load):
                raise OSError("Invalid model file: " + modelfile_load)
            elif not os.access(modelfile_load,os.R_OK):
                raise OSError("No read permission for model file: " + modelfile_load)
            model.load_state_dict(torch.load(modelfile_load))
    #load dataset
    dataset = classifier.TreeSpeciesDataset(inp,species_list)
    test_amount = len(dataset)//5
    train_dataset, test_dataset = classifier.safe_train_test_split(dataset,test_amount)
    print((len(train_dataset),len(test_dataset)))
    #create dataloader
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    #do training
    if weighted:
        weights = torch.Tensor(dataset.distrib)/sum(dataset.distrib)
        losser = torch.nn.CrossEntropyLoss(weight = weights)
    else:
        losser = torch.nn.CrossEntropyLoss()
    if accuracy:
        losses,accu = interTrain(model,optim,train_data,image_corrector,augments,losser,epoch,limit,test_data)
    else:
        losses = interTrain(model,optim,train_data,image_corrector,augments,losser,epoch,limit,None)
    #save_model
    if len(sys.argv) >= 5:
        torch.save(model.state_dict(), modelfile_save)
        #save results
        name = ".".split(os.path.basename(modelfile_save))[0]
        with open(os.path.join(save_dir,name + ".csv"), "a") as test_file:
            test_file.write("loss," + ",".join([str(val) for val in losses]) + "\n")
            if accuracy:
                test_file.write("accu," + ",".join([str(val) for val in accu]) + "\n")

if __name__ == "__main__":
    main()