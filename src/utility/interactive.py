import matplotlib.pyplot as plt
from src.utility.data_info import get_task_type

def show_data(loader, class_names=None):

    num_workers = loader.num_workers
    loader.num_workers = 0 # Changing this temporarily lets us exist enumeration faster; it hangs temporarily otherwise

    if get_task_type(loader) == 'segmentation': 
        class_names = None # Can't do this for segmentation masks

    for i, batch in enumerate(loader):
        image_batch, label_batch = batch
        image, label = image_batch[0], label_batch[0] # Get a single image from the batch
        print(image.shape, label) # View the batch shapes
        plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
        if class_names is not None: 
            plt.title(class_names[label])
        plt.axis(False)
        plt.show()
        cmd = input("Quit? Any Input for Yes else [ENTER]\n")
        if len(cmd) == 0:
            break
    loader.num_workers = num_workers # Change this back

# interactive_data_showcase = False
    # if interactive_data_showcase:
    #     show_data(train_loader, class_names)