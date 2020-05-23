from torchvision.models import resnet50
import torch
from torch.autograd import Variable
import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import json
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_label_encoding():
    with open('imagenet_class_index.json') as f:
        data = json.load(f)
        labels_json = data
        labels = {int(idx):label for idx, label in labels_json.items()}
    
    return labels

def get_real_image_vector():
    image = Image.open('Elephant2.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(image)[None,:,:,:]
    img_variable = Variable(image_tensor, requires_grad=True)
    img_variable.data = image_tensor

    return img_variable

def visualize_image_tensor(title, img_tensor):
    img = img_tensor.squeeze(0)
    img = img.mul(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1)).add(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1)).detach().numpy()
    plt.imshow(np.transpose(img , (1,2,0)))
    plt.title(title)
    plt.show()

def fgsm(x, y, targeted=False, eps=0.2, alpha=0.03, iteration=35):
        # initialize
        adv_image = Variable(x.data, requires_grad=True)

        for i in range(iteration):
            h_adv = model(adv_image)

            # early termination condition
            if targeted:
                cost = F.cross_entropy(h_adv, y)
                if (h_adv.squeeze().max(0)[1]==y[0]).item():
                    print("Fooled netowrk in "+str(i)+" iterations")
                    break
            else:
                cost = -F.cross_entropy(h_adv, y)

            model.zero_grad()
            cost.backward()

            # compute image gradient wrt loss
            x_grad = alpha * torch.sign(adv_image.grad.data)
            adv_temp = adv_image.data - x_grad
            total_grad = adv_temp - x
            total_grad = torch.clamp(total_grad, -eps, eps)

            # updating image tensor with gradient 
            adv_image_update = x + total_grad
            adv_image = Variable(adv_image_update.data, requires_grad=True)

        real_label = model(x).squeeze().max(0)[1].item()
        adv_label = model(adv_image).squeeze().max(0)[1].item()

        return adv_image, adv_label, real_label

if __name__ == "__main__":
    # compute label encodings
    label_encoding = get_label_encoding()

    # get real image converted to tensor
    image_variable = get_real_image_vector()

    # load model
    model = resnet50(pretrained=True).eval()

    # print real label
    real_label = model(image_variable).squeeze().max(0)[1].item()
    print("Real label - "+label_encoding[real_label][1])

    # get perturbed image and aversory label, run fgsm for 1 iteration
    real_target = Variable(torch.LongTensor([101]), requires_grad=False)
    adv_image, adv_label, real_label = fgsm(image_variable, real_target, iteration=1)
    print("Adversory image label - " + label_encoding[adv_label][1])
    visualize_image_tensor("Adversory image", adv_image)

    # run fgsm to get target prediction of bullet_train
    adv_target = Variable(torch.LongTensor([466]), requires_grad=False)
    adv_image, adv_label, real_label = fgsm(image_variable, adv_target, targeted=True)
    print("Adversory image label - " + label_encoding[adv_label][1])

    # visualize images
    visualize_image_tensor("Original image", image_variable)
    visualize_image_tensor("Adversory image", adv_image)