import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data_loader import get_loader
from model import CT2captionModel
#from transformer import CT2captionModel
from util.caption_utils import save_checkpoint, load_checkpoint, print_examples
def train():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
           # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="D:/data/brain/preprocessed_data_v2/fourier_cut_cropped_img/cropped_img/",
        annotation_file="D:/data/brain/captions.txt",
        transform=transform,
      #  num_workers = 2
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    #하이퍼파라미터
    dim_size = 64
    vocab_size = len(dataset.vocab)
    num_heads = 4
    num_decoder_layers = 6
    dropout_p = 0.1
    in_channels= 1
    patch_size = 2
    img_size = 16
    depth = 3
    learning_rate = 3e-4
    num_epochs = 30


    step = 0

    #initialize model , loss etc
    model = CT2captionModel(vocab_size,dim_size,num_heads,num_decoder_layers,dropout_p,in_channels,patch_size,img_size,depth).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\checkpoint_coco_30.pth.tar"),model,optimizer)

    model.train()

    start_token = torch.tensor([1], dtype=torch.long)

    for epoch in range(num_epochs):

        print_examples(model,start_token,device,dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)

        for idx,(imgs,captions,_) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.permute(1,0).to(device)

            y_input = captions[:,:-1]
            y_expected = captions[:,1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            outputs = model(imgs,y_input,tgt_mask)

            outputs = outputs.permute(1,2,0)
            loss = criterion(outputs,y_expected)

            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epochs",epoch,"Training loss", loss.item())

        # for idx,(imgs,captions,_) in enumerate(train_loader):
        #     src = imgs.to(device)
        #     tgt = captions.permute(1,0).to(device)
        # 
        #     tgt_input = tgt[:-1,:]
        # 
        #     src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(tgt, tgt_input)
        # 
        #     logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        # 
        #     optimizer.zero_grad()
        # 
        #     tgt_out = tgt[1:, :]
        #     loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        #     loss.backward()
        # 
        #     optimizer.step()
        #     losses += loss.item()
        # 
        # print("epochs", epoch, "Training loss", loss.item())

if __name__ == "__main__":
    train()
