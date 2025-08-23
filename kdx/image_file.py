import fitz
from PIL import Image, ImageDraw
import os
from random import randint as rint

from coords_file import CoordsConverter
from detect_file import is_wanted_line




#------------------------task-0------------------------

def get_png_image(pdf_path, png_output_path, dpi=300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        pix.save(png_output_path)
    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pil_img


def cut_png(img, crop_size=(352, 352), stride=(128, 128)):
    width, height = img.size
    tile_w, tile_h = crop_size
    stride_w, stride_h = stride

    cut_images = []
    for top in range(0, height - tile_h + 1, stride_h):
        for left in range(0, width - tile_w + 1, stride_w):
            box = (left, top, left + tile_w, top + tile_h)
            cut_images.append(((left, top), img.crop(box)))
    
    return cut_images





#------------------------task-1------------------------

def draw_qq_in_png(circles, draw_png_path, radius=6):
    Image.MAX_IMAGE_PIXELS = 2000000000
    with Image.open(draw_png_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for cx, cy in circles:
            draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline="red",
            width=6
            )
        img.save(draw_png_path)
    print(f'drew {len(circles)//3} circles in {os.path.basename(draw_png_path)} ({len(circles)} actually drawn)')



def get_lines(pdf_path, features, line_txt_path, CoordsConverter=CoordsConverter):
    with fitz.open(pdf_path) as doc:
        lines = []
        single_lines = []
        page = doc.load_page(0)
        drawings = page.get_drawings()
        coords_convert = CoordsConverter(page.rect)

        for draw in drawings:
            line_app = False
            if not is_wanted_line(draw, features):
                continue
            if len(draw['items']) == 1:
                line_app = True
            for item in draw['items']:
                if item[0]!='l':
                    continue
                x0, y0 = item[1]
                x1, y1 = item[2]
                x0, y0 = coords_convert.convert_pdf_to_png(x0,y0,page.rect.height)
                x1, y1 = coords_convert.convert_pdf_to_png(x1,y1,page.rect.height)
                lines.append((x0,y0,x1,y1))
                if line_app:
                    single_lines.append((x0,y0,x1,y1))
        with open(line_txt_path, 'w', encoding='utf-8') as linetxt:
            for line in lines:
                linetxt.write(f'{line}\n')
        return lines

def draw_lines(lines, draw_png_path):
    Image.MAX_IMAGE_PIXELS = 2000000000
    with Image.open(draw_png_path) as img:
        draw = ImageDraw.Draw(img)

        for line in lines:
            x0,y0,x1,y1 = line
            draw.line([x0,y0,x1,y1],fill=(rint(0, 80),rint(0,80),rint(180,255)),width=1)

        img.save(draw_png_path)
    print(f'drew {len(lines)} lines in {os.path.basename(draw_png_path)}')





#------------------------task-2-------------------------

def draw_and_diff_kdx(gangangs, final_png_path, not_kdx, all_lines):
    Image.MAX_IMAGE_PIXELS = 2000000000
    with Image.open(final_png_path) as img:
        gangang_rest = {}
        gangangs_2up = set()
        kdx_2up = set()
        draw = ImageDraw.Draw(img)
        kdx_c, gg_c = 0, 0

        for kdx, gg in gangangs.items():
            kx0,ky0,kx1,ky1 = kdx
            gg90, gg45 = gg
            
            if len(gg90)>=2 or len(gg45)>=2:
                kdx_c+=1
                gangangs_2up.update(tuple(gg90))
                gangangs_2up.update(tuple(gg45))
                kdx_2up.add(tuple(kdx))
                draw.line([(kx0, ky0), (kx1, ky1)], fill='purple', width=5)

                for x0,y0,x1,y1 in gg90:
                    gg_c+=1
                    draw.line([x0,y0,x1,y1],fill='green',width=3)
                for x0,y0,x1,y1 in gg45:
                    gg_c+=1
                    draw.line([x0,y0,x1,y1],fill='brown',width=2)
            else:
                gangang_rest[kdx]=gg

        possible_exggs = [i 
        for i in not_kdx 
            if i not in gangangs_2up]

        possible_exkdx = [i 
        for i in all_lines
            if i not in kdx_2up and i not in gangangs_2up]

        img.save(final_png_path)
    print(f'[{str(os.path.basename(final_png_path)).upper()}]: drew {kdx_c} kdx and {gg_c} ggs')
    return gangang_rest, possible_exkdx, possible_exggs



#------------------------task-3-------------------------

def draw_exkdx_n_exggs(gangangs, png_path):
    Image.MAX_IMAGE_PIXELS = 2000000000
    with Image.open(png_path) as img:
        draw = ImageDraw.Draw(img)
        cc_c, gg_c = 0, 0

        for kdx, gg in gangangs.items():
            for kx0,ky0,kx1,ky1 in kdx:
                cc_c += 1
                draw.line([kx0,ky0,kx1,ky1],fill='orange',width=3)

            for x0,y0,x1,y1 in gg:
                gg_c+=1
                draw.line([x0,y0,x1,y1],fill='green',width=3)

        img.save(png_path)
    print(f'[{str(os.path.basename(png_path)).upper()}]: drew {cc_c} ccs and {gg_c} ggs')


