import torch
import torch.linalg


def color_code_images(imgs: torch.Tensor, threshold: float):
    """Return color codes of each image in `images`.

    Parameters
    ----------
    imgs : Tensor
        Float 3-ch images whose shapes are [N, 3, H, W]

    Returns
    -------
    color_code : Tensor
        Color codes for each images. shape=[N]
    """
    assert imgs.dim() == 4, "imgs must be a 4d tensor"

    # Binarize input with channels kept
    binary_imgs = (imgs > threshold).to(torch.uint8)
    # Binary to int
    code_pixels = binary_imgs * torch.tensor([1, 2, 4], dtype=torch.uint8)\
                                     .view(-1, 3, 1, 1)
    code_pixels = code_pixels.sum(dim=1)
    # Attach labels to each pixel indicating the image that the pixel belongs to
    code_pixels_with_id = (code_pixels
                           + 8 * torch.arange(len(imgs)).view(-1, 1, 1))
    code_imgs_with_id = torch.bincount(code_pixels_with_id.ravel(),
                                       minlength=8 * len(imgs))
    # Vote
    vote_imgs = code_imgs_with_id.view(-1, 8)
    code_imgs = torch.argmax(vote_imgs[:, 1:], dim=1)
    return code_imgs

def color_code_images2(imgs: torch.Tensor, threshold: float):
    assert imgs.dim() == 4, "imgs must be a 4d tensor"

    with torch.no_grad():
        # set 0 to background
        imgs = torch.where(imgs > threshold, imgs, torch.zeros_like(imgs))
        # calc norm of color vector for all pixels
        norm_imgs = torch.sqrt(imgs.square().sum(dim=1,keepdims=True))
        # normalization
        normalized_imgs = imgs/(norm_imgs+1e-5)
        # count non-background pixels
        cnt_nonzeros = norm_imgs.sum(dim=1).count_nonzero(dim=2).sum(dim=1)
        cnt = torch.stack( (cnt_nonzeros, cnt_nonzeros, cnt_nonzeros) ,dim=1)
        # calc mean of color vectors in figure 
        normalized_imgs = (normalized_imgs.sum(dim=3).sum(dim=2))
        normalized_imgs = normalized_imgs/(cnt+0.1)
    return normalized_imgs

def color_invariance(x: torch.Tensor, y: torch.Tensor, threshold: float = .1):
    """Compute color invariance between batches of images `x` and `y`.
    Note that this function returns a scalar, which is different
    from the behavior of `shape_invariance()`.

    Parameters
    ----------
    x, y: Tensor
        Image tensors that must have the same shape [N, C, H, W].
    threshold: float
        Threshold for binarization. Default = .5

    Returns
    -------
    color_invariance: float
        1.0 if all the images have the same color, and
        0.0 if all the images have different colors.
    """
    #xcolor = color_code_images(x, threshold)
    #ycolor = color_code_images(y, threshold)
    #num_same = torch.count_nonzero(xcolor == ycolor )

    norm_x = color_code_images2(x, threshold)
    norm_y = color_code_images2(y, threshold)
    #return float(num_same) / len(x)
    return float((norm_x*norm_y).sum(dim=1).mean())


def shape_invariance(x: torch.Tensor, y: torch.Tensor, threshold: float = .3):
    """Compute shape invariance between batches of images `x` and `y`.

    Parameters
    ----------
    x, y : Tensor
        Image tensors that must have the same shape [N, C, H, W].
    threshold : float
        Threshold of binarization

    Returns
    -------
    shape_invariance: Tensor
        1D Tensor.
    """
    def binarize(x: torch.Tensor, threshold: float):
        return torch.any(x > threshold, dim=1).to(torch.float)

    def normalize(x: torch.Tensor):
        return x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6)

    # Binarize
    binary_x = binarize(x, threshold).flatten(start_dim=1)
    binary_y = binarize(y, threshold).flatten(start_dim=1)
    binary_x = normalize(binary_x)
    binary_y = normalize(binary_y)
    similarity = torch.diag(
        torch.matmul(binary_x,
                     torch.transpose(binary_y, 0, 1))
    )
    return float(similarity.mean())
