import shutil
import os

srcs = ["/root/autodl-tmp/ouhang/programming/steel-detection-research-phaseI/data/cubitic/P0703-d-GROUND FLOOR POUR 1 TOP REINFORCEMENT PLAN-B.pdf",    #probably qq0
"/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/Stronghold/S7.06_ LEVEL 1 SLAB TOP REINFORCEMENT PLAN Rev.D.pdf",
"/root/autodl-tmp/ouhang/programming/steel-detection-research-phaseI/data/cubitic/S0362-4-GROUND FLOOR TOP REINFORCEMENT PLAN- PART 2.pdf",
"/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/CUBITIC/P0713-d-GROUND FLOOR POUR 2 TOP REINFORCEMENT PLAN-A.pdf",
"/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/Stronghold/S7.08-LEVEL-3-SLAB-REINFORCEMENT-PLANS-Rev.E.pdf",
"/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/CUBITIC/P0702-d-GROUND FLOOR POUR 1 BOTTOM REINFORCEMENT PLAN-B.pdf",    #qq5
"/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/TTM/19141-S016-B_L2-BOTTOM REINFORCEMENT.pdf",
"/root/autodl-tmp/ouhang/programming/steel-detection-research-phaseI/data/PTWORKS/Test-3(PTWorks).pdf"]

srcs = ["/root/autodl-tmp/ouhang/data/original_pdf/I I Dimension Drawings/CUBITIC/P0702-d-GROUND FLOOR POUR 1 BOTTOM REINFORCEMENT PLAN-B.pdf"]

dst = '/root/autodl-tmp/hansen/kdx/pdfsorces/'
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

for i,src in enumerate(srcs):
    dst_file = f'1234567.pdf'
    shutil.copy(src,dst_file)