
from snaptools.snap import dump_read
import os
import pandas as pd

dir = "../data/mini_atlas/ATAC_MOp_EckerRen/snap/"

s = "CEMBA171207_3C"

dump_read(snap_file=os.path.join(dir, s + ".snap"),
          output_file=os.path.join(dir, s + ".fragments.bed"),
          buffer_size=10**5,
          barcode_file=None,
          tmp_folder=os.path.join(dir, "tmp"),
          overwrite=True)

dt = pd.read_csv(os.path.join(dir, s + ".fragments.bed"),
                 header=None,
                 sep="\t")
dt[0] = dt[0].apply(lambda x: x[2:-1])
dt = dt.groupby(dt.columns.tolist(),as_index=False).size()
dt = dt.sort_values([0, 1, 2, 3])
dt.to_csv(os.path.join(dir, s + ".fragments.sort.bed"),
          header=False, sep="\t", index=False)


# =================== from terminal
# sort, compress, and index
# # sort -k1,1 -k2,2n CEMBA171206_3C.fragments.bed | uniq -c > CEMBA171206_3C.fragments.sort.bed
# bgzip -@ 8 CEMBA171206_3C.fragments.sort.bed
# tabix -p bed CEMBA171206_3C.fragments.sort.bed.gz
#
# # clean up
# rm fragments.bed
#
# # take a look at the output
# gzip -dc fragments.sort.bed.gz | head