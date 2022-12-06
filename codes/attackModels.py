import torch
func_call = 0

'''TODO2: make this into a separate file'''
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # we have to make func_call global
    # otherwise it can't be accessed
    # get the sign of fgsm
    sign_data_grad = data_grad.sign()

    '''debuging'''
    global func_call
    if (func_call == 0):
        print('data_grad_shape_example: ', data_grad.shape)
        print('data_grad_sign_shape: ', sign_data_grad.shape)
    func_call = func_call + 1
    '''debuging'''

    # get perturbated image
    perturbed_image = image + epsilon*sign_data_grad
    # in case image are out of range after using perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # return adversarial image
    return perturbed_image
