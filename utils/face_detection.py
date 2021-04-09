from pathlib import Path
import cv2
import numpy as np
from copy import deepcopy


def template_matching(original_image, detect='face', return_coords=False):
	template_path = Path().cwd().joinpath('templates').joinpath('{}.png'.format(detect))
	template_image = cv2.imread(str(template_path), 0)
	h, w = template_image.shape

	image = deepcopy(original_image)
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	res = cv2.matchTemplate(grayscale_image, template_image, cv2.TM_SQDIFF_NORMED)
	_, _, top_left, _ = cv2.minMaxLoc(res)
	bottom_right = (top_left[0] + w, top_left[1] + h)
	if return_coords:
		return top_left, bottom_right

	cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
	return image


def viola_jones(original_image, detect='face&eyes'):
	def draw_detected(img, detected_list, color=(0, 255, 0)):
		for (column, row, width, height) in detected_list:
			cv2.rectangle(
				img,
				(column, row),
				(column + width, row + height),
				color,
				1)
	pretrained_path = Path().cwd().joinpath('pretrained')

	image = deepcopy(original_image)
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if detect in ['face', 'face&eyes']:
		face_detector = cv2.CascadeClassifier(str(pretrained_path.joinpath('haarcascade_frontalface_default.xml')))
		detected_faces = face_detector.detectMultiScale(grayscale_image)
		draw_detected(image, detected_faces, (0, 0, 255))

	if detect in ['eyes', 'face&eyes']:
		eyes_detector = cv2.CascadeClassifier(str(pretrained_path.joinpath('haarcascade_eye.xml')))
		detected_eyes = eyes_detector.detectMultiScale(grayscale_image)
		draw_detected(image, detected_eyes)

	return image


def get_symmetry_line(face):
	h, w = face.shape[0], face.shape[1]
	border = max(w // 4, 5)

	min_distance = 1e15
	best_line = border
	for line in range(border, w - border):
		distances = []
		max_shift = min(line, w - line)
		for shift in range(1, max_shift):
			left_column = face[:, line - shift]
			right_column = face[:, line + shift]
			distances.append(np.abs(left_column - right_column).mean())
		distance = np.array(distances).mean()
		if distance < min_distance:
			min_distance = distance
			best_line = line

	return best_line


def get_symmetry_lines(face):
	nose_width = 5

	central_line = get_symmetry_line(face)

	left_face_part = face[:, 0:central_line-nose_width]
	right_face_part = face[:, central_line+nose_width:]

	left_line = get_symmetry_line(left_face_part)
	right_line = get_symmetry_line(right_face_part)

	return left_line, central_line, central_line+right_line


def symmetry_lines(original_image):
	image = deepcopy(original_image)
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	top_left, bottom_right = template_matching(image, detect='face', return_coords=True)
	face = grayscale_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
	l, c, r = get_symmetry_lines(face)
	l += top_left[0]
	c += top_left[0]
	r += top_left[0]

	h, w = image.shape[0], image.shape[1]

	# left line
	cv2.line(
		image,
		(l, h // 4),
		(l, 3 * h // 4),
		(0, 255, 0),
		1)

	# central line
	cv2.line(
		image,
		(c, h // 4),
		(c, 3 * h // 4),
		(0, 0, 255),
		1)

	# right line
	cv2.line(
		image,
		(r, h // 4),
		(r, 3 * h // 4),
		(0, 255, 0),
		1)

	return image


if __name__ == '__main__':
	pass
