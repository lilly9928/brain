import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from data_loader import get_loader
import torch.nn.functional as F
from tqdm import tqdm
from ImageCaption_model import CNNtoRNN
from PIL import Image


# Parameters

checkpoint_pth = 'D:/github/brain/Imagecaption/my_checkpoint_cnn.pth'
#word_map_file = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_loader, dataset = get_loader(
    root_folder="D:/data/brain/preprocessed_data_v2/fourier_cut_cropped_img/cropped_img/",
    annotation_file="D:/data/brain/captions.txt",
    transform=transform,
    #num_workers=2
)
# 하이퍼파라미터
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 2
learning_rate = 3e-4
num_epochs = 30

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load(checkpoint_pth)['state_dict'])
model.eval()


def evaluate():
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    loader, dataset = get_loader(
        root_folder="D:/data/brain/preprocessed_data_v2/fourier_cut_cropped_img/cropped_img/",
        annotation_file="D:/data/brain/captions.txt",
        transform=transforms.Compose([transform]),
        batch_size=1
    )

    wordmap = dataset.vocab

    allcaption =dict()
    idx = list()
    for i in range(0,len(dataset.imgs)):
        name = dataset.imgs[i]
        if name == dataset.imgs[i]:
            idx.append(i)
            allcaption[name] = idx
            idx = list()

    references = list()
    hypotheses = list()

    i = 0
    # For each image
    for _,img_name in enumerate(tqdm(allcaption)):
        i = i+1
        if i == 1000:
            break
        image = transform(
            Image.open(f"D:/data/brain/preprocessed_data_v2/fourier_cut_cropped_img/cropped_img/{img_name}").convert("L")).unsqueeze(0)
        captions=[]
        for idx in allcaption[img_name]:
            captions.append(loader.dataset.captions[idx])

        # Move to GPU device, if available
        image = image.to(device)
        starttoken =torch.tensor([1], dtype=torch.long).to(device)

        # Encode
        outputs = model.caption_images(image,wordmap)
        # References
        img_captions = list(map(lambda ref: ref.split(), captions)) # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in outputs if w not in {'<SOS>', '<EOS>', '<PAD>','<UNK>'}])

        total_bleu4=0
        total_bleu3 = 0
        total_bleu2 = 0
        total_bleu1 = 0

        #Calculate BLEU-4 scores
        total_bleu4 += corpus_bleu(references, hypotheses,weights=(0, 0, 0, 1))
        total_bleu3 += corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
        total_bleu2 += corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
        total_bleu1 += corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
       # total_bleu4 += corpus_bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), list(outputs.split()), weights=(0, 0, 0, 1))

    bleu4=total_bleu4%len(allcaption)
    bleu3 = total_bleu3 % len(allcaption)
    bleu2 = total_bleu2 % len(allcaption)
    bleu1=total_bleu1%len(allcaption)



   # bleu4 = corpus_bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), list(outputs.split()), weights=(0, 0, 0, 1))

    return print(bleu4,bleu3,bleu2,bleu1)


if __name__ == '__main__':
    evaluate()
    #print("\nBLEU-4 score @ is %.4f." % (evaluate()))
