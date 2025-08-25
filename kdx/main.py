import os
import shutil
from ultralytics import YOLO
from pathlib import Path

from coords_file import *
from image_file import *
from detect_file import *



if __name__ == '__main__':
    pdf_dir = '/root/autodl-tmp/hansen/kdx/pdfsorces'
    orig_png_dir = '/root/autodl-tmp/hansen/kdx/pngsorces'
    draw_png_dir = '/root/autodl-tmp/hansen/kdx/draw_png'
    final_png_dir = '/root/autodl-tmp/hansen/kdx/final_png'
    final_png2_dir = '/root/autodl-tmp/hansen/kdx/final_png2'
    qq_txt_dir = '/root/autodl-tmp/hansen/kdx/qq_txt'
    line_txt_dir = '/root/autodl-tmp/hansen/kdx/line_txt'
    kdx_txt_dir = '/root/autodl-tmp/hansen/kdx/kdx_txt'
    restkdx_txt_dir = '/root/autodl-tmp/hansen/kdx/restkdx_txt'
    exkdx_txt_dir = '/root/autodl-tmp/hansen/kdx/exkdx_txt'
    exggs_txt_dir = '/root/autodl-tmp/hansen/kdx/exggs_txt'
    model = YOLO('/root/autodl-tmp/hansen/kdx/best.pt')
    features = ['items', 'type', 'rect', 'color', 'width', 'dashes']
    lens = ((33,25,25), 55)    #min_len依次是kdx_gg90mid, kdx_gg45和kdx_gg90

    task = float('inf')
    if True:    #task和文件夹创建
        if os.path.exists(restkdx_txt_dir):
            task = 3
            shutil.copytree(final_png_dir, final_png2_dir)
        elif os.path.exists(line_txt_dir):    #findgg,find_extend
            task = 2
            shutil.copytree(draw_png_dir,final_png_dir)
            os.makedirs(restkdx_txt_dir)
            os.makedirs(exkdx_txt_dir)
            os.makedirs(exggs_txt_dir)
        elif os.path.exists(qq_txt_dir):
            task = 1    #drawqq, findkdx
            os.makedirs(line_txt_dir)
            os.makedirs(kdx_txt_dir)
            shutil.copytree(orig_png_dir, draw_png_dir)
        else:
            task = 0    #cutpic, findqq
            os.makedirs(orig_png_dir)
            os.makedirs(qq_txt_dir)



    for file_idx in range(len(list(Path(pdf_dir).glob("qq*.pdf")))):
        if not file_idx in [0,1,3,4,5,6,7]:    #0,3,5,6,7
            continue
        pdf_path = f'{pdf_dir}/qq{file_idx}.pdf'
        orig_png_path = f'{orig_png_dir}/qq{file_idx}.png'
        draw_png_path = f'{draw_png_dir}/qq{file_idx}.png'
        final_png_path = f'{final_png_dir}/qq{file_idx}.png'
        final_png2_path = f'{final_png2_dir}/qq{file_idx}.png'
        qq_txt_path = f'{qq_txt_dir}/qq{file_idx}.txt'
        line_txt_path = f'{line_txt_dir}/qq{file_idx}.txt'
        kdx_txt_path = f'{kdx_txt_dir}/qq{file_idx}.txt'
        restkdx_txt_path = f'{restkdx_txt_dir}/qq{file_idx}.txt'
        exkdx_txt_path = f'{exkdx_txt_dir}/qq{file_idx}.txt'
        exggs_txt_path = f'{exggs_txt_dir}/qq{file_idx}.txt'

        if True:    #根据task调用函数
            if task == 0:
                #cutpic
                pil_png = get_png_image(pdf_path,orig_png_path)
                cut_imgs = cut_png(pil_png)
                #findqq
                circles = predict_qq_with_yolo(cut_imgs, model, qq_txt_path)

            elif task == 1:
                #drawqq
                circles = get_qq(qq_txt_path)
                draw_qq_in_png(circles, draw_png_path)
                #findkdx
                lines = get_lines(pdf_path, features, line_txt_path)
                kdx_lines = find_kdx(lines, circles, kdx_txt_path)
                draw_lines(lines, draw_png_path)

            elif task == 2:
                #findgg
                not_kdx, kdx_lines, all_lines = seperate_kdx_and_rest(line_txt_path, kdx_txt_path, lens)
                gangangs = get_gangangs(not_kdx, kdx_lines, lens=lens)
                basekdx, exkdx, exggs = draw_and_diff_kdx(gangangs, final_png_path, not_kdx, all_lines, kdx_lines)
                renew_txt(((restkdx_txt_path,basekdx), (exkdx_txt_path,exkdx), (exggs_txt_path,exggs)))
                #如果重新运行的话记得改一下get_gangangs把45°也加一个角度限制

            elif task == 3:
                basekdx_dict, not_kdx, kdx_lines = get_rexgangangs(restkdx_txt_path, exggs_txt_path, exkdx_txt_path)
                gangangs = find_kdx_by_cc_simple(kdx_lines, not_kdx)
                draw_exkdx_n_exggs(gangangs, final_png2_path)

            else:
                print('Traceback (most recent call last):\n     File "/root/autodl-tmp/envs/ouhang/lib/python3.10/site-packages/PIL/ImageFile.py", line 249, in _save\n        you_got_tricked = "lol"\nWellItsAnError: your fault')
                break




'''    #规划
crop_size = 352
stride = 128

把pdf转化成png
把png切成小图
把小图用YOLO的best.pt检测出里面有没有圆/圈圈
把圆在小图里的坐标转化成原pdf里的坐标

找出在pdf中的所有线
这些线中：
	找到穿过圆的线
		这条穿过圆的线必须穿过/挨着另外两个东西
			这个东西可以是线也可以是X样式的两条线
	并标出
这种线也就是跨度线
'''

'''    #反正是yolo的一个什么东西
result:

ultralytics.engine.results.Boxes object with attributes:

cls: tensor([0.], device='cuda:0')
conf: tensor([0.9529], device='cuda:0')
data: tensor([[252.7673,  95.4621, 270.8840, 113.4936,   0.9529,   0.0000]], device='cuda:0')
id: None
is_track: False
orig_shape: (352, 352)
shape: torch.Size([1, 6])
xywh: tensor([[261.8257, 104.4779,  18.1167,  18.0315]], device='cuda:0')
xywhn: tensor([[0.7438, 0.2968, 0.0515, 0.0512]], device='cuda:0')
xyxy: tensor([[252.7673,  95.4621, 270.8840, 113.4936]], device='cuda:0')
xyxyn: tensor([[0.7181, 0.2712, 0.7696, 0.3224]], device='cuda:0')

0: 352x352 1 quanquan, 4.5ms
Speed: 0.4ms preprocess, 4.5ms inference, 0.7ms postprocess per image at shape (1, 3, 352, 352)
'''

'''    #pdf遍历的时候的dict结构
{'items': 
    [('l', Point(1061.0399169921875, 508.47998046875), Point(1060.919921875, 508.3599853515625)), 
    ('l', Point(1060.919921875, 508.3599853515625), Point(1060.7999267578125, 508.239990234375)), 
    ('l', Point(1060.7999267578125, 508.239990234375), Point(1060.679931640625, 508.3599853515625)), 
    ('l', Point(1060.679931640625, 508.3599853515625), Point(1060.5599365234375, 508.47998046875)), 
    ('l', Point(1060.5599365234375, 508.47998046875), Point(1060.679931640625, 508.5999755859375)), 
    ('l', Point(1060.679931640625, 508.5999755859375),Point(1060.7999267578125, 508.719970703125)), 
    ('l', Point(1060.7999267578125, 508.719970703125), Point(1060.919921875, 508.5999755859375)), 
    ('l', Point(1060.919921875, 508.5999755859375), Point(1061.0399169921875, 508.47998046875))], 
'closePath': False, 
'type': 'fs', 
'even_odd': False, 
'fill_opacity': 1.0, 
'fill': (0.0, 0.0, 0.0), 
'rect': Rect(1060.5599365234375, 508.239990234375, 1061.0399169921875, 508.719970703125), 
'seqno': 10393, 
'layer': 
'x-GND|Hatch Over', 
'stroke_opacity': 1.0, 
'color': (0.0, 0.0, 0.0), 
'width': 0.0, 
'lineCap': (1, 1, 1),
'lineJoin': 0.11999999731779099, 
'dashes': '[] 0'
}
'''

'''    #更全的features
features = [
    'items',
    'closePath',
    'type',
    'even_odd',
    'fill_opacity',
    'fill',
    'rect',
    'seqno',
    'layer',
    'stroke_opacity',
    'color',
    'width',
    'lineCap',
    'lineJoin',
    'dashes'
]
'''

"""    #老的找杠杠 更新跨度线的代码和配套的画跨度线的代码
    for gx0,gy0,gx1,gy1 in not_kdx:
        cx,cy = (gx0+gx1)/2, (gy0+gy1)/2
        for kx0,ky0,kx1,ky1 in kdx_lines:
            
            x_diff = abs(abs(kx0-cx)+abs(kx1-cx)-abs(kx0-kx1))
            y_diff = abs(abs(ky0-cy)+abs(ky1-cy)-abs(ky0-ky1))
            if x_diff+y_diff<tol:
                gangangs.append(((gx0,gy0,gx1,gy1),(kx0,ky0,kx1,ky1)))
                break
            '''
            for kx, ky in [(kx0,ky0),(kx1,ky1)]:
                if abs(cy-ky)+abs(cx-kx)<tol:
                    gangangs.append((gx0,gy0,gx1,gy1))
                else:
                    continue
                break'''
    print(f'found {len(gangangs)} gangangs and {len(kdx_lines)} kdx lines')
    return gangangs


def get_kdx_with_gg(kdx_lines, gangangs, tol=2):
    gg_midpoints = [((x0 + x1)/2, (y0 + y1)/2) for x0, y0, x1, y1 in gangangs]
    kdxs = []

    for kx0, ky0, kx1, ky1 in kdx_lines:
            
        ps = sum(
                abs(abs(kx0-mx)+abs(kx1-mx)-abs(kx0-kx1))+
                abs(abs(ky0-my)+abs(ky1-my)-abs(ky0-ky1)) 
                < tol for mx, my in gg_midpoints
                )

        if ps == 2:
            kdxs.append((kx0,ky0,kx1,ky1))
        else:
            continue

    return kdx_lines

--------|-------

def draw_kdx(kdx, final_png_path):
    Image.MAX_IMAGE_PIXELS = 2000000000
    with Image.open(final_png_path) as img:
        draw = ImageDraw.Draw(img)

        for x0,y0,x1,y1 in kdx:
            draw.line([x0,y0,x1,y1],fill='green',width=4)
    
        img.save(final_png_path)
"""

'''    #老的_build_graph_kdx 更简单但更慢（没觉得慢多少）
    def _endpoint_coincide(s1,s2,eps_pos):
        _pt = lambda seg,i: (seg[0],seg[1]) if i==0 else (seg[2],seg[3])
        _dist = lambda a,b: math.hypot(a[0]-b[0], a[1]-b[1])
        return any(_dist(_pt(s1,i), _pt(s2,j)) <= eps_pos for i in (0,1) for j in (0,1))

    def _colinear_overlap(s1,s2,eps_ang,eps_line,eps_olap):
        def _proj_interval(seg, axis):
            p0 = (seg[0],seg[1]); p1 = (seg[2],seg[3])
            s0 = p0[0]*axis[0] + p0[1]*axis[1]
            s1 = p1[0]*axis[0] + p1[1]*axis[1]
            return (s0,s1) if s0 <= s1 else (s1,s0)
        
        def _dist_point_to_line(pt, seg):
            x0,y0,x1,y1 = seg
            px,py = pt
            vx,vy = x1-x0, y1-y0
            wx,wy = px-x0, py-y0
            v2 = vx*vx + vy*vy
            if v2 == 0.0:
                return math.hypot(px-x0, py-y0)
            return abs(wx*vy - wy*vx) / math.sqrt(v2)

        def _angdiff(a,b):
            d = abs(a-b) % 180.0
            return 180.0 - d if d > 90.0 else d

        def _ang(seg):
            x0,y0,x1,y1 = seg
            return math.degrees(math.atan2(y1-y0, x1-x0)) % 180.0

        a1 = _ang(s1); a2 = _ang(s2)
        if _angdiff(a1,a2) > eps_ang: 
            return False
        if _dist_point_to_line((s1[0],s1[1]),s2) > eps_line: return False
        if _dist_point_to_line((s1[2],s1[3]),s2) > eps_line: return False
        if _dist_point_to_line((s2[0],s2[1]), s1) > eps_line: return False
        if _dist_point_to_line((s2[2],s2[3]), s1) > eps_line: return False
        ax = (math.cos(math.radians((a1+a2)/2.0)), math.sin(math.radians((a1+a2)/2.0)))
        s0,s1v = _proj_interval(s1, ax)
        t0,t1v = _proj_interval(s2, ax)
        return (min(s1v,t1v) - max(s0,t0)) >= eps_olap


    def _build_graph_kdx(kdx_lines, eps_pos=3.0, eps_ang=5.0, eps_line=2.0, eps_olap=3.0):
        n = len(kdx_lines)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                s1, s2 = kdx_lines[i], kdx_lines[j]
                if _endpoint_coincide(s1,s2,eps_pos) or _colinear_overlap(s1,s2,eps_ang,eps_line,eps_olap):
                    adj[i].append(j)
                    adj[j].append(i)
        return adj
'''


