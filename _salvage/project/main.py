from experiment import Experiment
import sys

if __name__ == "__main__":

    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)

    best_iou_score = exp.run()
    print(f"Best IoU score: {best_iou_score}")
    
    test_loss, test_iou, test_acc, image_outputs, image_labels = exp.test()
    print(f"Test Loss is {test_loss}")
    print(f"Test IoU is {test_iou}")
    print(f"Test Pixel acc is {test_acc}")
    
    