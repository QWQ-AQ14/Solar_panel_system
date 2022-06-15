import numpy as np
import vounwarp.losa.loadersaver as io
import vounwarp.post.postprocessing as post
from PIL import Image

# Load image
img_path = r"E://xlq//图片集//项目资料//红外素材20220110//永昌一期//11、15、19区//DJI_0242.jpg"
mat0 = io.load_image(img_path)
output_base = "figs/"
mat = np.asarray(Image.open(img_path), dtype=np.float32)
# Import distortion coefficients
(xcenter, ycenter, list_fact) = io.load_metadata_txt("figs/coefficients.txt")

for i in range(mat.shape[-1]):
    mat[:, :, i] = post.unwarp_image_backward(mat[:, :, i], xcenter, ycenter, list_fact)
io.save_image(output_base + "/Sol0_1st_color_correction.png", mat)


(height, width) = mat0.shape
mat0 = mat0 / np.max(mat0)

# Estimated forward model
xcenter = width / 2.0 + 110.0
ycenter = height / 2.0 - 20.0
list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
list_coef = np.asarray([1.0, 4.0, 5.0, 17.0, 3.0])
list_ffact = list_pow * list_coef

# Calculate parameters of a backward model from the estimated forward model
list_hor_lines = []
for i in range(20, height-20, 50):
    list_tmp = []
    for j in range(20, width-20, 50):
        list_tmp.append([i - ycenter, j - xcenter])
    list_hor_lines.append(list_tmp)
Amatrix = []
Bmatrix = []
list_expo = np.arange(len(list_ffact), dtype=np.int16)
for _, line in enumerate(list_hor_lines):
    for _, point in enumerate(line):
        xd = np.float64(point[1])
        yd = np.float64(point[0])
        rd = np.sqrt(xd * xd + yd * yd)
        ffactor = np.float64(np.sum(list_ffact * np.power(rd, list_expo)))
        if ffactor != 0.0:
            Fb = 1 / ffactor
            ru = ffactor * rd
            Amatrix.append(np.power(ru, list_expo))
            Bmatrix.append(Fb)
Amatrix = np.asarray(Amatrix, dtype=np.float64)
Bmatrix = np.asarray(Bmatrix, dtype=np.float64)
list_bfact = np.linalg.lstsq(Amatrix, Bmatrix, rcond=1e-64)[0]

# Apply distortion correction
corrected_mat = post.unwarp_image_backward(mat0, xcenter, ycenter, list_bfact)
io.save_image(output_base + "/after.png", corrected_mat)
io.save_image(output_base + "/before.png", mat0)
io.save_metadata_txt(output_base + "/coefficients.txt", xcenter, ycenter, list_bfact)

