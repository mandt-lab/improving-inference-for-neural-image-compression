checkpoint_dir = './checkpoints'

# compression / evaluation settings
write_tfci_for_eval = False
eval_batch_num_pixels = 1e7  # num pixels in the batch; corresponding to 10 1000x1000 images, using 0.03GB memory (conversion from number of pixels to bytes: #bytes = #pixels * 24 / 8)


def get_eval_batch_size(num_pixels_per_image):
    return round(eval_batch_num_pixels / num_pixels_per_image)

# for comparing our discretization method against others (only for the non-bits-back version)
save_opt_record = False

