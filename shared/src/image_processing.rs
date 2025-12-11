use crate::item::{HEIGHT, WIDTH};
use image::{GrayImage, Luma};
pub fn rasterize_strokes(
    strokes: &Vec<Vec<[f32; 3]>>
) -> [[f32; WIDTH]; HEIGHT] {
    let stroke_thickness: f32 = 1.0;
    // 1. Compute bounding box
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for stroke in strokes {
        for &[x, y, _t] in stroke {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    let width = max_x - min_x;
    let height = max_y - min_y;

    if width <= 0.0 || height <= 0.0 {
        return [[0.0; WIDTH]; HEIGHT];
    }

    // 2. Compute scale (preserve aspect ratio)
    let scale = (WIDTH as f32 / width).min(HEIGHT as f32 / height);

    // 3. Initialise image
    let mut img = [[0.0f32; WIDTH]; HEIGHT];

    // Helper for safe pixel write
    let mut plot = |x: isize, y: isize, intensity: f32| {
        if x >= 0 && x < WIDTH as isize && y >= 0 && y < HEIGHT as isize {
            let (x, y) = (x as usize, y as usize);
            img[y][x] = img[y][x].max(intensity);
        }
    };

    // 4. Draw lines with Bresenham
    for stroke in strokes {
        for window in stroke.windows(2) {
            let [x0, y0, _] = window[0];
            let [x1, y1, _] = window[1];

            // Transform to pixel space
            let mut px0 = ((x0 - min_x) * scale) as isize;
            let mut py0 = ((y0 - min_y) * scale) as isize;
            let px1 = ((x1 - min_x) * scale) as isize;
            let py1 = ((y1 - min_y) * scale) as isize;

            // Bresenham’s line algorithm
            let dx = (px1 - px0).abs();
            let dy = -(py1 - py0).abs();
            let sx = if px0 < px1 { 1 } else { -1 };
            let sy = if py0 < py1 { 1 } else { -1 };
            let mut err = dx + dy;

            loop {
                // Draw pixel with optional thickness
                plot(px0, py0, 1.0);

                // Thickness (circle brush)
                let r = stroke_thickness as isize;
                for ox in -r..=r {
                    for oy in -r..=r {
                        if ox * ox + oy * oy <= r * r {
                            plot(px0 + ox, py0 + oy, 0.99);
                        }
                    }
                }

                if px0 == px1 && py0 == py1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    px0 += sx;
                }
                if e2 <= dx {
                    err += dx;
                    py0 += sy;
                }
            }
        }
    }

    img
}

pub fn save_image(img: &[[f32; WIDTH]; HEIGHT], label: String) {
    let mut im = GrayImage::new(WIDTH as u32, HEIGHT as u32);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            // Convert 0.0-1.0 float to 0-255 u8
            let intensity = (img[y][x] * 255.0).clamp(0.0, 255.0) as u8;
            im.put_pixel(x as u32, y as u32, Luma([255 - intensity]));
        }
    }

    im.save(label + ".png").unwrap();
}