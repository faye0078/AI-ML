import torch
import os
from tqdm import tqdm
from classify_model import make_model
from classify_data import make_dataloaders
from classify_utils.config import obtain_classify_args
from classify_utils.validation import validate
def main():
    args = obtain_classify_args()
    model = make_model(args.model)

    model = model.cuda()

    model_path = '../saved_model/{}.pth.tar'.format(args.model)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    train_loader, test_loader = make_dataloaders(args.dataset)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    best_val = 0.0
    acc = validate(0, test_loader, model, criterion)
    best_val = acc
    for epoch in range(100):


        train_loss = 0.0
        model.train()

        tbar = tqdm(train_loader)
        for i, sample in enumerate(tbar):
            image = sample[0].cuda()
            label = sample[1].cuda()

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, label)
            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            loss.backward()
            optimizer.step()
        print('epoch{} train loss: '.format(epoch) + str(train_loss))

        acc = validate(epoch, test_loader, model, criterion)
        print('current best acc: ' + str(best_val))

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), '../saved_model/{}.pth.tar'.format(args.model))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

