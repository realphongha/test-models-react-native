export const drawPoint = (image, w, h, xRaw, yRaw, color, radius) => {
  // draws a plus sign to a RGB array image [h, w, 3]
  let x = Math.round(xRaw);
  let y = Math.round(yRaw);
  image[y * w * 3 + x * 3] = color[0];
  image[y * w * 3 + x * 3 + 1] = color[1];
  image[y * w * 3 + x * 3 + 2] = color[2];
  for (let i = 1; i < radius; i++) {
    for (let j of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (j[0] >= 0 && j[0] < w && j[1] >= 0 && j[1] < h) {
        image[j[1] * w * 3 + j[0] * 3] = color[0];
        image[j[1] * w * 3 + j[0] * 3 + 1] = color[1];
        image[j[1] * w * 3 + j[0] * 3 + 2] = color[2];
      }
    }
  }
}

export const drawBox = (image, xmin, ymin, xmax, ymax, w, h,
  color, radius) => {
  for (let i = xmin; i <= xmax; i++) {
    drawPoint(image, w, h, i, ymin, color, radius);
  }
  for (let i = xmin; i <= xmax; i++) {
    drawPoint(image, w, h, i, ymax, color, radius);
  }
  for (let i = ymin; i <= ymax; i++) {
    drawPoint(image, w, h, xmin, i, color, radius);
  }
  for (let i = ymin; i <= ymax; i++) {
    drawPoint(image, w, h, xmax, i, color, radius);
  }
}

export const drawKeypoints = (image, points, w, h, color, radius) => {
  for (let point of points) {
    let [x, y] = point;
    drawPoint(image, w, h, x, y, color, radius);
  }
}

export const drawKeypointsColor = (image, points, w, h, color, radius) => {
  for (let i = 0; i < points.length; i++) {
    let [x, y] = points[i];
    drawPoint(image, w, h, x, y, color[i], radius);
  }
}