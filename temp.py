# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting

x_dimension = 2721
y_dimension = 1956

stars_file = get_pkg_data_filename("stars.fits")
fits.info(stars_file)
stars_data = fits.getdata(stars_file, ext=0)
original_image = stars_data


def star_center_locator(sample_pixel_coordinate, data, dimention):
    x_bar = 0
    y_bar = 0
    tot_intensity = 0
    for i in range(dimention):
        for j in range(dimention):
            intensity = data[i - dimention // 2 + sample_pixel_coordinate[1]][
                j - dimention // 2 + sample_pixel_coordinate[0]]
            tot_intensity += intensity
    for i in range(dimention):
        for j in range(dimention):
            intensity = data[i - dimention // 2 + sample_pixel_coordinate[1]][
                j - dimention // 2 + sample_pixel_coordinate[0]]
            x_bar += (j - dimention // 2 + sample_pixel_coordinate[0]) * intensity / tot_intensity
            y_bar += (i - dimention // 2 + sample_pixel_coordinate[1]) * intensity / tot_intensity
    x_bar = int(round(x_bar))
    y_bar = int(round(y_bar))
    return ([x_bar, y_bar])


def distance(xy1, xy2):
    return (np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2))


def disk_intensity(center_coordinate, radius, data):
    count = 0
    n = 0
    box = np.zeros((2 * radius + 1, 2 * radius + 1))
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            coordinate = [center_coordinate[0] + x - radius, center_coordinate[1] + y - radius]
            if distance(coordinate, center_coordinate) - radius <= 0:
                pixel = data[center_coordinate[1] - radius + y][center_coordinate[0] - radius + x]
                count += pixel
                box[y][x] = pixel
                n += 1
    std = np.std(box)
    return count, n, std, box


def star_radius(center_coordinate, r_initial, radius, data, number):
    r_line = []
    SNR_list = []
    sky_intensity = disk_intensity(star_1, radius + 25, stars_data)[0] - \
                    disk_intensity(star_1, radius + 15, stars_data)[0]
    sky_number = disk_intensity(star_1, radius + 15, stars_data)[1] - disk_intensity(star_1, radius + 5, stars_data)[1]
    sky = sky_intensity / sky_number
    for i in range(radius - r_initial):
        r_line.append(i + r_initial)
        info = disk_intensity(center_coordinate, i + r_initial, data)
        Sum_minus_sky = info[0] - sky * info[1]
        Noise = np.sqrt(info[0] * 1 + info[1] * 1 ** 2)
        SNR = Sum_minus_sky * 1 / Noise
        SNR_list.append(SNR)
    stellar_radius = r_line[SNR_list.index(max(SNR_list))]
    intensity = disk_intensity(center_coordinate, stellar_radius, data)[0]
    SNR_star = max(SNR_list)
    plt.plot(r_line, SNR_list)
    plt.title("SNR againts radius of Star Number " + str(number))
    plt.xlabel("radius")
    plt.ylabel("SNR")
    plt.show()
    plt.imshow(disk_intensity(center_coordinate, stellar_radius, data)[3], cmap="gray")
    plt.title("Star Number " + str(number))
    plt.show()
    print("Star Radius of Star Number " + str(number) + " = ", stellar_radius)
    print("Star Intensity of Star Number " + str(number) + " = ", intensity)
    print("SNR of Star Number " + str(number) + " = ", SNR_star)
    return stellar_radius, intensity, SNR_star


def magnitude_calculator(known_magnitude, known_intensity, known_SNR, intensity, SNR):
    magnitude = known_magnitude - 2.5 * np.log10(intensity / known_intensity)
    sigma = 1.0875 * np.sqrt(((1 / SNR) ** 2 + (1 / known_SNR) ** 2))
    return magnitude, sigma


star_1 = star_center_locator([426, y_dimension - 1167], stars_data, 40)
star_2 = star_center_locator([716, y_dimension - 554], stars_data, 40)
star_3 = star_center_locator([680, y_dimension - 480], stars_data, 40)
star_4 = star_center_locator([640, y_dimension - 390], stars_data, 40)
star_5 = star_center_locator([1263, y_dimension - 1330], stars_data, 30)

data_set_1 = star_radius(star_1, 0, 30, stars_data, 1)
data_set_2 = star_radius(star_2, 0, 30, stars_data, 2)
data_set_3 = star_radius(star_3, 0, 30, stars_data, 3)
data_set_4 = star_radius(star_4, 0, 30, stars_data, 4)
data_set_5 = star_radius(star_5, 0, 10, stars_data, 5)

print("Magnitude for star number 2 = ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_2[1], data_set_2[2])[0], " ± ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_2[1], data_set_2[2])[1])
print("Magnitude for star number 3 = ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_3[1], data_set_3[2])[0], " ± ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_3[1], data_set_3[2])[1])
print("Magnitude for star number 4 = ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_4[1], data_set_4[2])[0], " ± ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_4[1], data_set_4[2])[1])
print("Magnitude for star number 5 = ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_5[1], data_set_5[2])[0], " ± ",
      magnitude_calculator(8.88, data_set_1[1], data_set_1[2], data_set_5[1], data_set_5[2])[1])
