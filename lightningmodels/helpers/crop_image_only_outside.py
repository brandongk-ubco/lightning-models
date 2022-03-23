def crop_image_only_outside(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    img = img.max(axis=2)
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return row_start, row_end, col_start, col_end

