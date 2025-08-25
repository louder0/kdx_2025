import math




#------------------------task-0------------------------

def get_qq(qq_txt_path):
    circles = []
    with open(qq_txt_path,'r',encoding='utf-8') as qqtxt:
        for line in qqtxt:
            cx, cy = tuple(map(float, line.rstrip('\n')[1:-1].split(',')))
            circles.append((cx,cy))
    return circles

class CoordsConverter:    #rect是pdf的page.rect, page_height是page.rect.height
    def __init__(self, rect):
        self.global_max_x = rect.x1
        self.global_max_y = rect.y1

    def convert_pdf_to_png(self, x_pdf, y_pdf, page_height, dpi=300):
        if self.global_max_x > self.global_max_y:
            x_png = x_pdf * dpi / 72
            y_png = y_pdf * dpi / 72
            return x_png, y_png
        else:
            x_png = (page_height-x_pdf) * dpi / 72
            y_png = (y_pdf) * dpi / 72  # PDF坐标系和PNG坐标系y轴方向相反
            return y_png, x_png

    def convert_png_to_pdf(self, x_png, y_png, page_height, dpi=300):
        if self.global_max_x > self.global_max_y:
            x_pdf = x_png * 72 / dpi
            y_pdf = y_png * 72 / dpi
            return x_pdf, y_pdf
        else:
            x_pdf = page_height - y_png * 72 / dpi
            y_pdf = x_png * 72 / dpi
            return x_pdf, y_pdf



#------------------------task-1-------------------------

def is_point_on_line(p1, p2, p, tol=1):
    x1, y1 = p1
    x2, y2 = p2
    px, py = p

    vx, vy = x2 - x1, y2 - y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:  # 退化为点
        return math.hypot(px - x1, py - y1) <= tol

    # 归一化投影参数 t（parametric t）
    t = ((px - x1)*vx + (py - y1)*vy) / seg_len2

    # 端点放宽：按线长折算一个很小的 eps
    eps = tol / math.sqrt(seg_len2)  # 例如 tol=1 像素，线越长 eps 越小
    if t < -eps or t > 1.0 + eps:
        return False

    # 垂直距离：| (P-A) × v | / |v|  （叉积 Cross product）
    cross = (px - x1)*vy - (py - y1)*vx
    perp_dist = abs(cross) / math.sqrt(seg_len2)

    return perp_dist <= tol
    '''
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p

    # 若线段退化为一个点（p1和p2重合），直接比较点与该点的距离
    if x1 == x2 and y1 == y2:
        dist = math.hypot(x3 - x1, y3 - y1)
        return dist <= tol

    # 计算向量AB和AP
    ABx, ABy = x2 - x1, y2 - y1
    APx, APy = x3 - x1, y3 - y1

    # 计算P在直线AB上的投影参数r
    dot = ABx * APx + ABy * APy           # 点积 AP·AB
    seg_len_sq = ABx * ABx + ABy * ABy    # |AB|^2
    r = dot / seg_len_sq

    # 判断投影点是否在线段内部
    if r < 0 or r > 1:
        return False

    # 计算垂足C坐标
    foot_x = x1 + r * ABx
    foot_y = y1 + r * ABy

    # 计算垂直距离并与tol比较
    dist = math.hypot(x3 - foot_x, y3 - foot_y)
    return dist <= tol'''

def find_kdx(lines, circles, kdx_txt_path, tol=2):
    kdx_lines = []
    for line in lines:
        x0,y0,x1,y1 = line
        for px, py in circles:
            if not is_point_on_line((x0,y0),(x1,y1),(px,py),tol):
                continue
            kdx_lines.append(line)
            break
    
    with open(kdx_txt_path,'w',encoding='utf-8') as kdx_txt:
        for kdx in kdx_lines:
            kdx_txt.write(f'{kdx}\n')

    return kdx_lines



#------------------------task-2-------------------------

def seperate_kdx_and_rest(line_txt_path, kdx_txt_path, lens):
    # 先把 kdx 的整行字符串放进 set
    with open(kdx_txt_path, 'r',encoding='utf-8') as f:
        kdx_set = {line.strip() for line in f if line.strip()}

    # 解析成浮点（一次）作为返回
    kdx_lines = [tuple(map(float, s[1:-1].split(','))) for s in kdx_set]

    all_lines = []
    not_kdx_lines = []
    with open(line_txt_path, 'r',encoding='utf-8') as f:
        for s in f:
            t = s.strip()
            if not t: 
                continue

            tline = tuple(map(float, t[1:-1].split(',')))
            all_lines.append(tline)

            if t in kdx_set:
                continue

            tlensq = (tline[0]-tline[2])**2+(tline[1]-tline[3])**2
            min_len, max_len = min(lens[0])**2, lens[1]**2
            if min_len <= tlensq <= max_len:
                not_kdx_lines.append(tline)
    print('===='*20)
    return not_kdx_lines, kdx_lines, all_lines

def renew_txt(paths):    #f'{k}:{v}'
    for path, data in paths:
        with open(path,'w',encoding='utf-8') as txt:
            if isinstance(data,list):
                for i in data:
                    txt.write(f'{i}\n')
            elif isinstance(data,dict):
                for k,v in data.items():
                    txt.write(f'{k}:{v}\n')



#------------------------task-3-------------------------





