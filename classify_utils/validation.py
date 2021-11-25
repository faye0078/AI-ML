import torch
from tqdm import tqdm

def validate(epoch, val_loader, model, criterion):

    tbar = tqdm(val_loader, desc='\r')
    test_loss = 0.0
    total = 0
    for i, sample in enumerate(tbar):
        image, label = sample[0], sample[1]
        image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, label)
        test_loss += loss.item()

        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        pred = torch.nn.functional.softmax(output, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total += 1
    print('epoch{} val loss: '.format(epoch) + str(test_loss))
    accuracy = total / (len(val_loader) * 128)
    print('epoch{} accuracy: '.format(epoch) + str(accuracy))
    return accuracy