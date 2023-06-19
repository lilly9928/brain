import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, start_token,device, dataset):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("D:/data/brain/test_image/1.jpg").convert("L")).unsqueeze(0)
    print("Example 1 CORRECT: Small SAH  right parietal lobe sulci Diffuse brain atrophy with ventriculomegaly Suggestive of small vessel disease  both cerebral white matter and basal ganglia ")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.example_images(test_img1.to(device), start_token.to(device),dataset.vocab,device)))
    test_img2 = transform(
        Image.open("D:/data/brain/test_image/2.jpg").convert("L")).unsqueeze(0)
    print("Example 2 CORRECT: Acute SDH along bilateral convexities and falx Associated SAH along the basal cisterns and cerebral sulci")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.example_images(test_img2.to(device), start_token.to(device),dataset.vocab,device)))
    test_img3 = transform(Image.open("D:/data/brain/test_image/3.jpg").convert("L")).unsqueeze(0)
    print("Example 3 CORRECT: Multi cisternal acue SAH  along basal sylvian ciserns and cortical cisterns Associated IVH hydrocephalus Skull vault is unremarkable Conclusion Acute SAH IVH  hydrocephalus")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.example_images(test_img3.to(device), start_token.to(device),dataset.vocab,device)))
    test_img4 = transform(Image.open("D:/data/brain/test_image/4.jpg")).unsqueeze(0)
    print("Example 4 CORRECT:Clinical information  trauma Acute SAH along the basal cisterns and both cerebral sulci  right left small acute SDH along both frontal convexities and falx Small amount of pneumocephalus noted in left side cavernous sinus and T S sinuses  rather likely IV related air than trauma induced Recommend   Clinical correlation")
    print("Example 4 OUTPUT: "+ " ".join(model.example_images(test_img4.to(device), start_token.to(device),dataset.vocab,device)))
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step