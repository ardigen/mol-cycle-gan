from random import randint, uniform

import numpy as np


class ImagePool:
    def __init__(self, pool_size=200):
        assert pool_size >= 0 and pool_size == int(pool_size),\
            "'pool_size' must be a positive integer or be equal to 0!"

        self.pool_size = pool_size
        self.images = []

    def query_over_images(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = uniform(0, 1)
                if p > 0.5:
                    random_id = randint(0, self.pool_size - 1)
                    replaced_image = self.images[random_id]
                    self.images[random_id] = image
                    return_images.append(replaced_image)
                else:
                    return_images.append(image)
        return_images = np.stack(return_images, axis=0)
        return return_images
