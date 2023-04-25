from new_main import *
from collections import defaultdict

def get_diagnosis(mask, pred):
    ref = True if torch.sum(mask).item() != 0 else False
    pre = True if torch.sum(pred).item() != 0 else False
    if ref and pre:             # true positive
        return 'tp'
    elif not ref and not pre:   # true negative
        return 'tn'
    elif ref and not pre:       # false negative
        return 'fn'
    else:                       # false positive
        print(torch.sum(pred).item(), end=',')
        return 'fp'        

if __name__ == "__main__":

    model_save_path = 'model.pth'
    model = unet.UNet(n_class=n_class).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))
    model.eval()
    
    org_sample_loader = get_dataloader(root, 'test', transforms=sample_transform, batch_size=1, shuffle=False)
    test_sample_loader = get_dataloader(root, 'test', transforms=valtest_transform, batch_size=1, shuffle=False)
    
    fncount = 0
    diagnosis = defaultdict(int)
    
    for (org, mask), (img, _) in zip(org_sample_loader, test_sample_loader):
        img = img.to(device)
        output = model(img)
        _, pred = torch.max(output, dim=1)
        diag = get_diagnosis(mask[0], pred[0])
        diagnosis[diag] += 1
        util.save_sample(np.array(org[0].cpu(), dtype=np.uint8), mask[0].cpu(), pred[0].cpu(), './samples/sample_'+str(fncount)+'.png')
        fncount += 1
        if fncount % 100 == 0:
            print(f"{fncount} images saved, diagnosis: {diagnosis}")

    print(diagnosis)        
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
