import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # 水平翻转本质上可以通过旋转180度(矩阵转置)再垂直翻转(矩阵行逆序排列)实现
        if flip_img:
            # H(height) x W(width) x C(channel), 注意排列的方法不是 C x H x W
            return np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        # shift ∈ [-3, 4). 初始位置从图的左上角开始，下和右为正. 超出原图的部分采用zero padding
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        img_pad = np.zeros_like(img)
        H, W = img.shape[0], img.shape[1]
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return img_pad
        img_pad[max(0, -shift_x):min(H - shift_x, H),
                max(0, -shift_y):min(W - shift_y, W), :] = img[max(0, shift_x):min(H + shift_x, H),
                                                               max(0, shift_y):min(W + shift_y, W), :]
        return img_pad
        ### END YOUR SOLUTION
