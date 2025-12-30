import numpy as np

def census_transform(img: np.ndarray,window_size: int = 2) ->np.ndarry:
    assert img.ndim ==2, "grayscale only"
    H,W = img.shape
    code = np.zeros((H,W),dtype = np.uint32)

    center = img.astype(np.uint32)

    bit = 0
    for dy in range(-window_size,window_size+1):
        for dx in range(-window_size,window_size+1):
            if dy == 0 and dx == 0:
                continue

            shifted = np.zeros((H,W),dtype = np.uint32)

            y0 = max(0,dy)
            y1 = min(H,H+dy)
            x0 = max(0,dx)
            x1 = min(W,W+dx)

            shifted[y0:y1,x0:x1] = center[y0-dy:y1-dy,x0-dx:x1-dx]

            b = (shifted < center).astype(np.uint32)

            code |= (b<<bit)

            bit += 1

            if bit > 48:
                break
        if bit > 48:
            break
    
    return code
